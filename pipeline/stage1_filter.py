import json
import logging
import math
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

BATCH_SIZE = 30  # compound-mini is request-bound (30 RPM) not token-bound (70K TPM) — larger
# batches cost nothing in TPM terms and cut call count (author correction, Phase 3).
RETRY_BATCH_SIZE = 5
CORPUS_CAP = 600
SAMPLE_SEED = 42

# Addition (author, Phase 3): thresholds checked after a full run, before Stage 2 ever spends
# quota on the survivors. A thin or lopsided corpus needs a look before proceeding.
MIN_SURVIVORS = 200
MAX_SOURCE_RATIO = 3.0

# Addition (author, Phase 3): 3,709 snippets at the old batch size of 20 was 186 calls against
# compound-mini's 250/day cap — no margin for retries, and most of the corpus gets discarded by
# the 600-cap below regardless. Presample down to ~1,300 first, weighted by expected signal
# density, before Stage 1 makes a single call. All YouTube comments are kept (highest density
# of purchase-hesitation language); Play Store is downsampled.
PRESAMPLE_AJIO_COUNT = 400
PRESAMPLE_MYNTRA_COUNT = 200

# Addition (author, Phase 3): abort before making any calls if the estimated request volume for
# this run can't possibly finish within a safe fraction of the daily cap. A run that cannot
# finish should fail before it starts, not partway through burning quota.
QUOTA_ABORT_THRESHOLD = 0.6


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def presample_corpus(snippets: list[dict], seed: int = SAMPLE_SEED) -> tuple[list[dict], dict]:
    """Weighted presample, run before any LLM call. Returns (sampled_snippets, report)."""
    rng = random.Random(seed)

    youtube = [s for s in snippets if s["source"] == "youtube"]
    ajio = [s for s in snippets if s["source"] == "play_store" and s.get("app") == "ajio"]
    myntra = [s for s in snippets if s["source"] == "play_store" and s.get("app") == "myntra"]
    categorized_ids = {s["id"] for s in youtube + ajio + myntra}
    other = [s for s in snippets if s["id"] not in categorized_ids]

    sampled_youtube = youtube  # keep all
    sampled_ajio = rng.sample(ajio, min(PRESAMPLE_AJIO_COUNT, len(ajio)))
    sampled_myntra = rng.sample(myntra, min(PRESAMPLE_MYNTRA_COUNT, len(myntra)))

    presampled = sampled_youtube + sampled_ajio + sampled_myntra + other

    report = {
        "youtube": {"pre": len(youtube), "post": len(sampled_youtube)},
        "play_store_ajio": {"pre": len(ajio), "post": len(sampled_ajio)},
        "play_store_myntra": {"pre": len(myntra), "post": len(sampled_myntra)},
        "other": {"pre": len(other), "post": len(other)},
        "total": {"pre": len(snippets), "post": len(presampled)},
    }

    log.info("=== Presample (before any Stage 1 call) ===")
    for label, counts in report.items():
        log.info(f"  {label}: {counts['pre']} -> {counts['post']}")

    return presampled, report


def estimate_quota(num_snippets: int, batch_size: int, model_env_var: str) -> dict:
    expected_calls = math.ceil(num_snippets / batch_size) if num_snippets else 0
    daily_limit = llm.DAILY_REQUEST_LIMITS[model_env_var]
    fraction = expected_calls / daily_limit if daily_limit else 1.0
    estimate = {
        "expected_calls": expected_calls,
        "daily_limit": daily_limit,
        "fraction": round(fraction, 4),
        "over_threshold": fraction > QUOTA_ABORT_THRESHOLD,
    }
    log.info(
        f"Quota estimate ({model_env_var}): {expected_calls} expected requests / "
        f"{daily_limit} daily cap = {fraction:.1%}"
    )
    return estimate


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
        raw_snippets = json.load(f)

    presampled, presample_report = presample_corpus(raw_snippets, seed=seed)

    quota_estimate = estimate_quota(len(presampled), BATCH_SIZE, "GROQ_MODEL_CHEAP")
    if quota_estimate["over_threshold"]:
        msg = (
            f"ABORTING before any Stage 1 calls: estimated {quota_estimate['expected_calls']} "
            f"requests is {quota_estimate['fraction']:.1%} of GROQ_MODEL_CHEAP's "
            f"{quota_estimate['daily_limit']}/day cap, over the {QUOTA_ABORT_THRESHOLD:.0%} threshold."
        )
        log.error(msg)
        return {
            "halted": True,
            "halt_reasons": [msg],
            "presample": presample_report,
            "quota_estimate": quota_estimate,
            "input_count": len(raw_snippets),
            "survivor_count": 0,
            "by_source": {},
            "dropped": [],
        }

    survivors, dropped = classify_snippets(presampled)

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

    input_by_source = Counter(s["source"] for s in presampled)
    survivor_by_source = Counter(s["source"] for s in survivors)

    log.info("=== Stage 1 survivor breakdown by source (of presampled corpus) ===")
    breakdown = {}
    for source in sorted(input_by_source):
        total = input_by_source[source]
        surv = survivor_by_source.get(source, 0)
        rate = surv / total if total else 0.0
        breakdown[source] = {"survived": surv, "of": total, "rate": round(rate, 4)}
        log.info(f"  {source}: {surv}/{total} survived ({rate:.1%})")
    log.info(f"  TOTAL: {len(survivors)}/{len(presampled)} survived ({len(survivors) / len(presampled):.1%})")

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
        "presample": presample_report,
        "quota_estimate": quota_estimate,
        "input_count": len(raw_snippets),
        "presampled_count": len(presampled),
        "survivor_count": len(survivors),
        "pre_sample_cap_survivor_count": pre_sample_count,
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
