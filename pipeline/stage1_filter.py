import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

import llm
from schema import RelevanceBatch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage1")

BATCH_SIZE = 20
RETRY_BATCH_SIZE = 5
CORPUS_CAP = 600
SAMPLE_SEED = 42

# Addition (author, Phase 3): thresholds checked after a full run, before Stage 2 ever spends
# quota on the survivors. A thin or lopsided corpus needs a look before proceeding.
MIN_SURVIVORS = 200
MAX_SOURCE_RATIO = 3.0


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _build_prompt(batch: list[dict]) -> str:
    lines = [f'{i + 1}. "{s["text"][:300]}"' for i, s in enumerate(batch)]
    return (
        "You are filtering user feedback for a product research pipeline. For each of the "
        f"following {len(batch)} snippets, determine if it is about saving, wishlisting, "
        "deferring, hesitating over, or abandoning a fashion/clothing purchase — not about "
        "delivery issues, app bugs, payment problems, customer service, or unrelated topics.\n\n"
        "Respond with JSON only, no prose, no markdown fences, matching this exact shape:\n"
        '{"relevant": [true, false, ...]}\n'
        f"The array must have exactly {len(batch)} elements, in the same order as the snippets "
        "below.\n\nSnippets:\n" + "\n".join(lines)
    )


def _classify_batch(batch: list[dict]) -> tuple[list[bool] | None, str | None]:
    """Returns (relevance_flags, None) on success, or (None, failure_reason)."""
    prompt = _build_prompt(batch)
    result = llm.call_cheap(prompt, expected_output_tokens=len(batch) * 6)
    if not result.ok:
        return None, "rate_limited" if result.rate_limited else "llm_error"
    try:
        parsed = RelevanceBatch.model_validate_json(_strip_fences(result.text))
    except ValidationError:
        return None, "malformed"
    if len(parsed.relevant) != len(batch):
        return None, "length_mismatch"
    return parsed.relevant, None


def classify_snippets(snippets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Core batching logic, no file I/O, no corpus cap, no halt check — reusable from both
    the CLI driver below and Streamlit's live paste-box path (ARCHITECTURE.md §2.1)."""
    survivors: list[dict] = []
    dropped: list[dict] = []

    for start in range(0, len(snippets), BATCH_SIZE):
        batch = snippets[start : start + BATCH_SIZE]
        relevance, failure = _classify_batch(batch)

        if relevance is not None:
            survivors.extend(s for s, is_relevant in zip(batch, relevance) if is_relevant)
            continue

        log.warning(f"Batch of {len(batch)} failed ({failure}), retrying at batch size {RETRY_BATCH_SIZE}")
        for sub_start in range(0, len(batch), RETRY_BATCH_SIZE):
            sub_batch = batch[sub_start : sub_start + RETRY_BATCH_SIZE]
            sub_relevance, sub_failure = _classify_batch(sub_batch)
            if sub_relevance is not None:
                survivors.extend(s for s, is_relevant in zip(sub_batch, sub_relevance) if is_relevant)
            else:
                for s in sub_batch:
                    dropped.append({"id": s["id"], "reason": sub_failure})
                log.error(
                    f"Sub-batch of {len(sub_batch)} still failed ({sub_failure}), dropping: "
                    f"{[s['id'] for s in sub_batch]}"
                )

    return survivors, dropped


def run(
    input_path: str = "data/raw_snippets.json",
    output_path: str = "data/filtered_snippets.json",
    corpus_cap: int = CORPUS_CAP,
    seed: int = SAMPLE_SEED,
) -> dict:
    with open(input_path, encoding="utf-8") as f:
        snippets = json.load(f)

    survivors, dropped = classify_snippets(snippets)

    pre_sample_count = len(survivors)
    if pre_sample_count > corpus_cap:
        rng = random.Random(seed)
        survivors = rng.sample(survivors, corpus_cap)
        log.info(f"Corpus cap: {pre_sample_count} survivors -> sampled down to {corpus_cap} (seed={seed})")
    else:
        log.info(f"Corpus cap: {pre_sample_count} survivors, under the {corpus_cap} cap, no sampling needed")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(survivors, f, ensure_ascii=False, indent=2)

    input_by_source = Counter(s["source"] for s in snippets)
    survivor_by_source = Counter(s["source"] for s in survivors)

    log.info("=== Stage 1 survivor breakdown by source ===")
    breakdown = {}
    for source in sorted(input_by_source):
        total = input_by_source[source]
        surv = survivor_by_source.get(source, 0)
        rate = surv / total if total else 0.0
        breakdown[source] = {"survived": surv, "of": total, "rate": round(rate, 4)}
        log.info(f"  {source}: {surv}/{total} survived ({rate:.1%})")
    log.info(f"  TOTAL: {len(survivors)}/{len(snippets)} survived ({len(survivors) / len(snippets):.1%})")

    if dropped:
        log.info(f"Dropped {len(dropped)} snippets due to persistent batch failures (see report['dropped'])")

    # Addition (author, Phase 3): stop before Stage 2 spends quota on a thin or lopsided corpus.
    halt_reasons = []
    if len(survivors) < MIN_SURVIVORS:
        halt_reasons.append(f"total survivors ({len(survivors)}) below the {MIN_SURVIVORS} minimum")

    ps = breakdown.get("play_store")
    yt = breakdown.get("youtube")
    if ps and yt and ps["rate"] > 0 and yt["rate"] > 0:
        ratio = max(ps["rate"], yt["rate"]) / min(ps["rate"], yt["rate"])
        if ratio > MAX_SOURCE_RATIO:
            halt_reasons.append(
                f"Play Store survival rate ({ps['rate']:.1%}) vs YouTube ({yt['rate']:.1%}) "
                f"differ by {ratio:.1f}x (over the {MAX_SOURCE_RATIO}x threshold)"
            )
    elif ps and yt and ps["rate"] == 0 and yt["rate"] > 0:
        halt_reasons.append("Play Store survival rate is 0% while YouTube is non-zero — maximally lopsided")
    elif ps and yt and yt["rate"] == 0 and ps["rate"] > 0:
        halt_reasons.append("YouTube survival rate is 0% while Play Store is non-zero — maximally lopsided")

    report = {
        "input_count": len(snippets),
        "survivor_count": len(survivors),
        "pre_sample_survivor_count": pre_sample_count,
        "by_source": breakdown,
        "dropped": dropped,
        "halted": bool(halt_reasons),
        "halt_reasons": halt_reasons,
    }

    if halt_reasons:
        log.error("STOPPING before Stage 2: " + "; ".join(halt_reasons))
    else:
        log.info("Stage 1 checks passed — safe to proceed to Stage 2")

    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "dropped"}, indent=2))
