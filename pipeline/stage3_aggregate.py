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


def opportunity_score(freq_pct: float, mean_intensity: float, addressable_share: float) -> float:
    weight = 1.0 if addressable_share > 0.5 else 0.3
    return freq_pct * mean_intensity * weight


def aggregate_records(records: list[dict]) -> dict:
    """In-memory core, no file I/O — reusable from Streamlit's live paste-box path
    (ARCHITECTURE.md §2.1)."""
    total = len(records)
    by_barrier = defaultdict(list)
    for r in records:
        by_barrier[r["barrier"]].append(r)

    barrier_table = []
    for barrier, recs in by_barrier.items():
        count = len(recs)
        freq_pct = (count / total * 100) if total else 0.0
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

    cross_save_intent = defaultdict(lambda: defaultdict(int))
    for r in records:
        cross_save_intent[r["barrier"]][r["save_intent"]] += 1

    cross_segment = defaultdict(lambda: defaultdict(int))
    for r in records:
        seg = r.get("segment_signal") or "unspecified"
        cross_segment[r["barrier"]][seg] += 1

    return {
        "total_records": total,
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
