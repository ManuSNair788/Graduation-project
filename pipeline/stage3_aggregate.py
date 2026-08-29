import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage3")

FORMULA = (
    "Opportunity Score = frequency_percent x mean_intensity x addressability_weight "
    "(addressability_weight = 1.0 if the majority of that barrier's snippets are "
    "addressable_without_money, else 0.3)"
)

OTHER_NOTE = (
    "Dominated by low-value \"please share the link\" comments that passed the Stage 1 "
    "relevance filter (they're about a fashion purchase) but carry no purchase-hesitation "
    "signal. Not a ranked opportunity area — it's a residual bucket, not something comparable "
    "or actionable against the named barriers below."
)

MIN_EVIDENCE_COUNT = 3

INSUFFICIENT_EVIDENCE_NOTE = (
    "Excluded from ranking: a category built on fewer than 3 snippets can be a single "
    "misclassified snippet rather than a real pattern (see Evidence tab). Not ranked "
    "alongside the barriers above, and not folded into their frequency_percent denominator."
)


def opportunity_score(freq_pct: float, mean_intensity: float, addressable_share: float) -> float:
    weight = 1.0 if addressable_share > 0.5 else 0.3
    return freq_pct * mean_intensity * weight


def aggregate_records(records: list[dict]) -> dict:
    """In-memory core, no file I/O — reusable from Streamlit's live paste-box path
    (ARCHITECTURE.md §2.1).

    Correction (author): "other" is a residual bucket, not an opportunity area — ranking it
    first (it was the largest category) made the output unusable for its stated purpose. It's
    excluded from the ranked barrier_table and reported separately instead; the named
    barriers' frequency_percent is recomputed against the non-"other" corpus, with both
    denominators stated so the numbers are traceable.

    Correction (author): a barrier built on fewer than MIN_EVIDENCE_COUNT snippets carries the
    same problem as "other" — one misclassified snippet (e.g. "please pin the link of white
    top" landing in choice_overload) produces a fully-formed-looking score with nothing real
    behind it. Applying the same principle: excluded from the ranked table, excluded from its
    frequency_percent denominator, reported separately with counts only (no score).
    """
    total = len(records)
    by_barrier = defaultdict(list)
    for r in records:
        by_barrier[r["barrier"]].append(r)

    other_records = by_barrier.pop("other", [])
    other_count = len(other_records)
    non_other_total = total - other_count

    thin_barriers = {b: recs for b, recs in by_barrier.items() if len(recs) < MIN_EVIDENCE_COUNT}
    for b in thin_barriers:
        by_barrier.pop(b)
    insufficient_evidence_total = sum(len(recs) for recs in thin_barriers.values())
    ranked_total = non_other_total - insufficient_evidence_total

    barrier_table = []
    for barrier, recs in by_barrier.items():
        count = len(recs)
        freq_pct = (count / ranked_total * 100) if ranked_total else 0.0
        mean_intensity = sum(r["intensity"] for r in recs) / count if count else 0.0
        addressable_share = (
            sum(1 for r in recs if r["addressable_without_money"]) / count if count else 0.0
        )
        score = opportunity_score(freq_pct, mean_intensity, addressable_share)
        barrier_table.append(
            {
                "barrier": barrier,
                "count": count,
                "frequency_percent": round(freq_pct, 2),
                "mean_intensity": round(mean_intensity, 2),
                "addressable_share": round(addressable_share, 2),
                "opportunity_score": round(score, 2),
            }
        )
    barrier_table.sort(key=lambda b: b["opportunity_score"], reverse=True)

    other_summary = {
        "count": other_count,
        "percent_of_corpus": round((other_count / total * 100) if total else 0.0, 2),
        "note": OTHER_NOTE,
    }

    insufficient_evidence = sorted(
        [{"barrier": b, "count": len(recs)} for b, recs in thin_barriers.items()],
        key=lambda b: -b["count"],
    )
    insufficient_evidence_summary = {
        "items": insufficient_evidence,
        "total_count": insufficient_evidence_total,
        "min_evidence_count": MIN_EVIDENCE_COUNT,
        "note": INSUFFICIENT_EVIDENCE_NOTE,
    }

    # Cross-cuts intentionally still include "other" — they're breakdowns, not rankings, and
    # showing the full picture (noise included) there is the honest choice.
    cross_save_intent = defaultdict(lambda: defaultdict(int))
    for r in records:
        cross_save_intent[r["barrier"]][r["save_intent"]] += 1

    cross_segment = defaultdict(lambda: defaultdict(int))
    for r in records:
        seg = r.get("segment_signal") or "unspecified"
        cross_segment[r["barrier"]][seg] += 1

    return {
        "total_records": total,
        "non_other_total": non_other_total,
        "ranked_total": ranked_total,
        "other_summary": other_summary,
        "insufficient_evidence_summary": insufficient_evidence_summary,
        "barrier_table": barrier_table,
        "barrier_x_save_intent": {b: dict(v) for b, v in cross_save_intent.items()},
        "barrier_x_segment_signal": {b: dict(v) for b, v in cross_segment.items()},
        "formula": FORMULA,
    }


def run(input_path: str = "data/extractions.json", output_path: str = "data/aggregates.json") -> dict:
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    aggregates = aggregate_records(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, ensure_ascii=False, indent=2)

    log.info(
        f"Stage3: aggregated {aggregates['total_records']} records into "
        f"{len(aggregates['barrier_table'])} barrier categories"
    )

    return aggregates


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["barrier_table"], indent=2))
