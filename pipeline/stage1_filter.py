import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

import llm
from schema import RelevanceBatch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage1")

MODEL_ENV_VAR = "GROQ_MODEL_STRONG"  # qwen/qwen3.8-27b — swapped in for the SECOND time today.
# groq/compound-mini hit its 250-requests/day cap first; its temporary replacement,
# openai/gpt-oss-20b (GROQ_MODEL_FILTER), then hit a 200,000-tokens/day cap
# (199,720/200,000 used) that DAILY_REQUEST_LIMITS never modeled — it was comfortably under its
# request-count budget the whole time. qwen has 2,000,000 tokens/day, ten times the headroom,
# and Stage 1 + Stage 2 combined fit safely inside both its request and token budgets today (see
# estimate_quota below). GROQ_MODEL_CHEAP and GROQ_MODEL_FILTER both stay defined and their
# llm.call_* functions unremoved, so either can be restored later with a one-line change here.

BATCH_SIZE = 10  # gpt-oss-20b is 8K TPM (not compound-mini's 70K) — a batch of 30 would exceed
# that per call. At 10, the existing TPM throttle becomes the binding constraint and paces the
# run; RPM (30) is no longer the tighter limit for this model (author correction, Phase 3).
RETRY_BATCH_SIZE = 5
CORPUS_CAP = 600
SAMPLE_SEED = 42

CHECKPOINT_PATH = "data/stage1_partial.jsonl"  # author correction, Phase 3, third pass: the
# real cost of the 2-hour stall was losing all completed work, not the stall itself. One JSON
# line per classified snippet, appended (and flushed) after every batch/sub-batch, so a crash,
# stall, or manual kill costs at most one batch, not the whole run.

HEARTBEAT_EVERY_N_BATCHES = 10  # author correction, Phase 3, third pass: makes a stall visible
# within minutes rather than hours.

# Addition (author, Phase 3): thresholds checked after a full run, before Stage 2 ever spends
# quota on the survivors. A thin or lopsided corpus needs a look before proceeding.
MIN_SURVIVORS = 200

# Correction (author, Phase 3, fourth pass): raised from 3.0 after two real runs both showed an
# ~8-9x Play Store/YouTube gap (4.8x, then 8.8x on a different presample) and a direct inspection
# of the rejected/accepted snippets (Phase 3, second pass) confirmed this is a structural property
# of the two sources, not a filter defect — Play Store reviews are overwhelmingly transactional
# (delivery, refunds, app crashes), while YouTube comments actually discuss purchase decisions.
# This check exists to catch a broken filter, and the filter is confirmed working. 10.0 still
# catches a genuine regression (e.g. one source suddenly returning ~0 survivors) without
# re-flagging the known, explained imbalance every run.
MAX_SOURCE_RATIO = 10.0

# Correction (author, Phase 3, third pass): Play Store survives at only ~1.8% (measured on the
# real run), so the full ~3,000-review pool costs roughly 300 calls to yield ~54 survivors —
# over half the day's call budget for a quarter of the output. Play Store is downsampled again,
# but to a higher, deliberately-chosen count (not the earlier 400/200 quota-conservation guess) —
# enough for source diversity without dominating the quota. All YouTube is still kept (8.7%
# survival — the corpus's main source of on-topic language).
PRESAMPLE_AJIO_COUNT = 700
PRESAMPLE_MYNTRA_COUNT = 300

# Addition (author, Phase 3): abort before making any calls if the estimated request volume for
# this run can't possibly finish within a safe fraction of the daily cap. A run that cannot
# finish should fail before it starts, not partway through burning quota.
QUOTA_ABORT_THRESHOLD = 0.6

# Conservative per-call token estimate for the quota projection below, informed by real
# openai/gpt-oss-20b usage today: roughly 120-150 successful Stage 1 batch calls (same prompt
# shape, batch size 10) consumed 199,720 tokens before hitting its daily cap — an observed
# average in the 1,300-1,700 tokens/call range. Rounded up for safety margin.
ESTIMATED_TOKENS_PER_CALL = 1600

# Rough pace observed during the one clean stretch of a real run (before the Retry-After bug
# degraded things) — used only to print an approximate runtime projection, not for any control
# decision.
OBSERVED_CALLS_PER_MIN = 9.8


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
    """Weighted presample — see PRESAMPLE_AJIO_COUNT/PRESAMPLE_MYNTRA_COUNT above for why these
    particular numbers. All YouTube kept; Play Store downsampled."""
    rng = random.Random(seed)

    youtube = [s for s in snippets if s["source"] == "youtube"]
    ajio = [s for s in snippets if s["source"] == "play_store" and s.get("app") == "ajio"]
    myntra = [s for s in snippets if s["source"] == "play_store" and s.get("app") == "myntra"]
    categorized_ids = {s["id"] for s in youtube + ajio + myntra}
    other = [s for s in snippets if s["id"] not in categorized_ids]

    sampled_ajio = rng.sample(ajio, min(PRESAMPLE_AJIO_COUNT, len(ajio)))
    sampled_myntra = rng.sample(myntra, min(PRESAMPLE_MYNTRA_COUNT, len(myntra)))

    presampled = youtube + sampled_ajio + sampled_myntra + other

    report = {
        "youtube": {"pre": len(youtube), "post": len(youtube)},
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
    """Checks BOTH daily budgets independently — request count and tokens. The request-only
    version of this function is what missed GROQ_MODEL_FILTER's 200K-tokens/day exhaustion: it
    was comfortably under its request-count budget the entire time it was failing."""
    expected_calls = math.ceil(num_snippets / batch_size) if num_snippets else 0

    daily_request_limit = llm.DAILY_REQUEST_LIMITS[model_env_var]
    request_fraction = expected_calls / daily_request_limit if daily_request_limit else 1.0

    daily_token_limit = llm.DAILY_TOKEN_LIMITS.get(model_env_var)
    expected_tokens = expected_calls * ESTIMATED_TOKENS_PER_CALL
    token_fraction = (expected_tokens / daily_token_limit) if daily_token_limit else 0.0

    estimate = {
        "expected_calls": expected_calls,
        "daily_request_limit": daily_request_limit,
        "request_fraction": round(request_fraction, 4),
        "expected_tokens": expected_tokens,
        "daily_token_limit": daily_token_limit,
        "token_fraction": round(token_fraction, 4) if daily_token_limit else None,
        "over_threshold": request_fraction > QUOTA_ABORT_THRESHOLD or token_fraction > QUOTA_ABORT_THRESHOLD,
    }

    log.info(
        f"Quota estimate ({model_env_var}): {expected_calls} requests / {daily_request_limit}/day "
        f"= {request_fraction:.1%}"
        + (
            f"; ~{expected_tokens} tokens / {daily_token_limit}/day = {token_fraction:.1%}"
            if daily_token_limit
            else " (no daily token cap)"
        )
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
    result = llm.call_strong(prompt, expected_output_tokens=len(batch) * 6)
    if not result.ok:
        return None, "rate_limited" if result.rate_limited else "llm_error"
    try:
        parsed = RelevanceBatch.model_validate_json(_strip_fences(result.text))
    except ValidationError:
        return None, "malformed"
    if len(parsed.relevant) != len(batch):
        return None, "length_mismatch"
    return parsed.relevant, None


def _load_checkpoint(checkpoint_path: str, id_lookup: dict) -> tuple[set, list[dict], list[dict]]:
    processed_ids: set = set()
    survivors: list[dict] = []
    dropped: list[dict] = []

    if not os.path.exists(checkpoint_path):
        return processed_ids, survivors, dropped

    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] not in id_lookup:
                continue  # stale entry from a different corpus/presample — safe to ignore
            if "relevant" in rec:
                # Only a genuine classification counts as done — skip it on resume.
                processed_ids.add(rec["id"])
                if rec["relevant"]:
                    survivors.append(id_lookup[rec["id"]])
            elif "dropped_reason" in rec:
                # A drop is a call failure (rate_limited/llm_error/malformed/length_mismatch),
                # not a content judgment — not marked processed, so it's retried fresh. Matters
                # concretely today: 5 snippets dropped when GROQ_MODEL_FILTER's token cap hit
                # would otherwise stay excluded forever, even after switching to a model that
                # can classify them fine.
                dropped.append({"id": rec["id"], "reason": rec["dropped_reason"]})

    # A later line may have succeeded on retry where an earlier one was dropped — don't report
    # the same snippet as both.
    dropped = [d for d in dropped if d["id"] not in processed_ids]

    return processed_ids, survivors, dropped


def classify_snippets(snippets: list[dict], checkpoint_path: str | None = None) -> tuple[list[dict], list[dict]]:
    """Core batching logic — reusable from both the CLI driver below and Streamlit's live
    paste-box path (ARCHITECTURE.md §2.1), which is why checkpointing is opt-in via
    `checkpoint_path` rather than always-on: the live path processes small ad hoc samples and
    has no business writing to the corpus-run checkpoint file.

    When `checkpoint_path` is given: loads already-processed snippet IDs from it and skips them
    (logging the resume position), then appends every new batch/sub-batch's verdicts to it as
    they complete — so a crash, stall, or manual kill costs at most one batch, not the whole run."""
    id_lookup = {s["id"]: s for s in snippets}
    survivors: list[dict] = []
    dropped: list[dict] = []
    processed_ids: set = set()

    if checkpoint_path:
        processed_ids, survivors, dropped = _load_checkpoint(checkpoint_path, id_lookup)
        if processed_ids:
            log.info(
                f"Resuming from checkpoint '{checkpoint_path}': {len(processed_ids)} already "
                f"processed ({len(survivors)} survivors, {len(dropped)} dropped so far)"
            )

    remaining = [s for s in snippets if s["id"] not in processed_ids]
    if processed_ids:
        log.info(f"{len(remaining)} snippets remaining out of {len(snippets)} total")

    checkpoint_file = None
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

    def _checkpoint_relevant(batch: list[dict], relevance: list[bool]) -> None:
        if not checkpoint_file:
            return
        for s, is_relevant in zip(batch, relevance):
            checkpoint_file.write(json.dumps({"id": s["id"], "relevant": is_relevant}) + "\n")
        checkpoint_file.flush()

    def _checkpoint_dropped(batch: list[dict], reason: str) -> None:
        if not checkpoint_file:
            return
        for s in batch:
            checkpoint_file.write(json.dumps({"id": s["id"], "dropped_reason": reason}) + "\n")
        checkpoint_file.flush()

    start_time = time.monotonic()
    total_batches = math.ceil(len(remaining) / BATCH_SIZE) if remaining else 0
    batch_num = 0

    try:
        for start in range(0, len(remaining), BATCH_SIZE):
            batch_num += 1
            batch = remaining[start : start + BATCH_SIZE]
            relevance, failure = _classify_batch(batch)

            if relevance is not None:
                survivors.extend(s for s, is_relevant in zip(batch, relevance) if is_relevant)
                _checkpoint_relevant(batch, relevance)
            else:
                log.warning(f"Batch of {len(batch)} failed ({failure}), retrying at batch size {RETRY_BATCH_SIZE}")
                for sub_start in range(0, len(batch), RETRY_BATCH_SIZE):
                    sub_batch = batch[sub_start : sub_start + RETRY_BATCH_SIZE]
                    sub_relevance, sub_failure = _classify_batch(sub_batch)
                    if sub_relevance is not None:
                        survivors.extend(s for s, is_relevant in zip(sub_batch, sub_relevance) if is_relevant)
                        _checkpoint_relevant(sub_batch, sub_relevance)
                    else:
                        for s in sub_batch:
                            dropped.append({"id": s["id"], "reason": sub_failure})
                        _checkpoint_dropped(sub_batch, sub_failure)
                        log.error(
                            f"Sub-batch of {len(sub_batch)} still failed ({sub_failure}), dropping: "
                            f"{[s['id'] for s in sub_batch]}"
                        )

            if batch_num % HEARTBEAT_EVERY_N_BATCHES == 0 or batch_num == total_batches:
                elapsed = time.monotonic() - start_time
                log.info(
                    f"Heartbeat: batch {batch_num}/{total_batches}, elapsed {elapsed:.0f}s, "
                    f"survivors so far {len(survivors)}"
                )
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    return survivors, dropped


def run(
    input_path: str = "data/raw_snippets.json",
    output_path: str = "data/filtered_snippets.json",
    checkpoint_path: str = CHECKPOINT_PATH,
    corpus_cap: int = CORPUS_CAP,
    seed: int = SAMPLE_SEED,
) -> dict:
    with open(input_path, encoding="utf-8") as f:
        raw_snippets = json.load(f)

    presampled, presample_report = presample_corpus(raw_snippets, seed=seed)

    quota_estimate = estimate_quota(len(presampled), BATCH_SIZE, MODEL_ENV_VAR)

    projected_runtime_min = quota_estimate["expected_calls"] / OBSERVED_CALLS_PER_MIN
    print()
    print("=== PRE-RUN PROJECTION ===")
    print(f"Presampled corpus size: {len(presampled)}")
    print(f"Expected requests: {quota_estimate['expected_calls']}")
    print(f"Request budget fraction ({MODEL_ENV_VAR}): {quota_estimate['request_fraction']:.1%}")
    if quota_estimate["daily_token_limit"]:
        print(
            f"Expected tokens: ~{quota_estimate['expected_tokens']} / "
            f"{quota_estimate['daily_token_limit']}/day = {quota_estimate['token_fraction']:.1%}"
        )
    else:
        print("No daily token cap for this model.")
    print(
        f"Projected runtime at ~{OBSERVED_CALLS_PER_MIN:.1f} calls/min: "
        f"{projected_runtime_min:.0f} min ({projected_runtime_min / 60:.1f}h)"
    )
    print()

    if quota_estimate["over_threshold"]:
        token_clause = (
            f" or ~{quota_estimate['expected_tokens']} tokens "
            f"({quota_estimate['token_fraction']:.1%} of its token cap)"
            if quota_estimate["daily_token_limit"]
            else ""
        )
        msg = (
            f"ABORTING before any Stage 1 calls: estimated {quota_estimate['expected_calls']} "
            f"requests ({quota_estimate['request_fraction']:.1%} of {MODEL_ENV_VAR}'s "
            f"{quota_estimate['daily_request_limit']}/day request cap){token_clause} exceeds the "
            f"{QUOTA_ABORT_THRESHOLD:.0%} threshold."
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

    survivors, dropped = classify_snippets(presampled, checkpoint_path=checkpoint_path)

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
