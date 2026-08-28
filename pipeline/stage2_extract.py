import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

import llm
from schema import ExtractionSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage2")

EXPECTED_OUTPUT_TOKENS = 300


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _build_prompt(snippet: dict) -> str:
    return (
        "Extract structured signal from this single piece of user feedback about a fashion "
        "e-commerce app (AJIO/Myntra). Use only information present in the text below — do not "
        "invent or assume any fact not stated.\n\n"
        f'Feedback: "{snippet["text"][:1000]}"\n\n'
        "Respond with JSON only, no prose, no markdown fences, matching this exact shape:\n"
        "{\n"
        '  "barrier": one of ["fit_size","price_uncertainty","styling_doubt","occasion_fit",'
        '"quality_doubt","choice_overload","trust_returns","stock_size_unavailable","other"],\n'
        '  "save_intent": one of ["purchase_intent","bookmarking","aspiration","comparison","unclear"],\n'
        '  "journey_stage": one of ["browse","saved","revisit","cart","checkout"],\n'
        '  "segment_signal": short free-text describing the user segment if inferable, or null,\n'
        '  "info_sought_outside_app": what they went elsewhere to find out, or null,\n'
        '  "workaround": what the user does instead of resolving the doubt in-app, or null,\n'
        '  "intensity": integer 1-5, how strongly this barrier blocks the purchase,\n'
        '  "addressable_without_money": true or false — could this plausibly be reduced without '
        "any discount, coupon, or monetary incentive,\n"
        '  "evidence": a verbatim quote from the feedback above, max 15 words\n'
        "}"
    )


def _to_record(snippet: dict, parsed: ExtractionSchema) -> dict:
    return {
        "id": snippet["id"],
        "source": snippet["source"],
        **parsed.model_dump(),
    }


def _process_snippet(snippet: dict) -> tuple[dict | None, str | None]:
    """Returns (extraction_record, None) on success, or (None, drop_reason).
    drop_reason in {"rate_limited", "llm_error", "malformed"}."""
    prompt = _build_prompt(snippet)

    raw = llm.call_strong(prompt, expected_output_tokens=EXPECTED_OUTPUT_TOKENS)
    if not raw.ok:
        return None, "rate_limited" if raw.rate_limited else "llm_error"
    try:
        parsed = ExtractionSchema.model_validate_json(_strip_fences(raw.text))
        return _to_record(snippet, parsed), None
    except ValidationError:
        pass  # escalate to the stronger model, per ARCHITECTURE.md §2.4

    raw_escalated = llm.call_synthesis(prompt, expected_output_tokens=EXPECTED_OUTPUT_TOKENS)
    if not raw_escalated.ok:
        return None, "rate_limited" if raw_escalated.rate_limited else "llm_error"
    try:
        parsed = ExtractionSchema.model_validate_json(_strip_fences(raw_escalated.text))
        return _to_record(snippet, parsed), None
    except ValidationError:
        return None, "malformed"


def run(
    input_path: str = "data/filtered_snippets.json",
    output_path: str | None = "data/extractions.json",
    limit: int | None = None,
) -> dict:
    with open(input_path, encoding="utf-8") as f:
        snippets = json.load(f)
    if limit is not None:
        snippets = snippets[:limit]

    records: list[dict] = []
    dropped: list[dict] = []

    for i, snippet in enumerate(snippets, start=1):
        record, reason = _process_snippet(snippet)
        if record is not None:
            records.append(record)
        else:
            dropped.append({"id": snippet["id"], "reason": reason})
            log.error(f"[{i}/{len(snippets)}] Dropped {snippet['id']} ({reason})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    rate_limited_dropped = sum(1 for d in dropped if d["reason"] == "rate_limited")
    malformed_dropped = sum(1 for d in dropped if d["reason"] == "malformed")
    other_dropped = sum(1 for d in dropped if d["reason"] not in ("rate_limited", "malformed"))

    log.info(
        f"Stage2: {len(records)} extracted, {len(dropped)} dropped "
        f"(rate_limited={rate_limited_dropped}, malformed={malformed_dropped}, other={other_dropped})"
    )

    return {
        "input_count": len(snippets),
        "records": records,
        "dropped": dropped,
        "rate_limited_dropped": rate_limited_dropped,
        "malformed_dropped": malformed_dropped,
        "other_dropped": other_dropped,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "dropped")}, indent=2))
