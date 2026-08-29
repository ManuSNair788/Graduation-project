import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import stage1_filter, stage2_extract, stage3_aggregate

st.set_page_config(page_title="AJIO Discovery Engine", layout="wide")

DATA_DIR = Path("data")


@st.cache_data
def load_cached():
    def _load(name):
        path = DATA_DIR / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return {
        "raw": _load("raw_snippets.json"),
        "filtered": _load("filtered_snippets.json"),
        "extractions": _load("extractions.json"),
        "aggregates": _load("aggregates.json"),
        "synthesis": _load("synthesis.json"),
        "stage1_report": _load("stage1_report.json"),
    }


def render_opportunity_table(aggregates: dict, key_prefix: str = "") -> None:
    if not aggregates or not aggregates.get("barrier_table"):
        st.info("No aggregate data available.")
        return

    total_records = aggregates.get("total_records")
    non_other_total = aggregates.get("non_other_total")
    other = aggregates.get("other_summary")

    if other and other["count"]:
        denom_note = (
            f"Ranked below: **{non_other_total} of {total_records}** classified snippets "
            f"(the {other['count']} classified as `other` — {other['percent_of_corpus']}% of the "
            f"corpus — are excluded from ranking and reported below the table)."
        )
        st.caption(denom_note)

    df = pd.DataFrame(aggregates["barrier_table"]).sort_values("opportunity_score", ascending=False)
    st.dataframe(df, width="stretch", hide_index=True)
    st.bar_chart(df.set_index("barrier")["opportunity_score"], horizontal=True)
    st.caption(
        "**Opportunity Score = frequency_percent x mean_intensity x addressability_weight** "
        "(addressability_weight = 1.0 if the majority of that barrier's snippets are "
        "addressable_without_money, else 0.3). frequency_percent for the 8 ranked barriers is "
        f"computed against **{non_other_total}** (total minus `other`), not the full "
        f"**{total_records}**."
    )

    if other and other["count"]:
        st.warning(
            f"**`other` — {other['count']} snippets ({other['percent_of_corpus']}% of the "
            f"{total_records}-snippet corpus) — excluded from the ranking above.**\n\n{other['note']}"
        )


def render_corpus_provenance(stage1_report: dict) -> None:
    if not stage1_report:
        st.info("No Stage 1 provenance data available.")
        return

    collected = stage1_report["input_count"]
    presampled = stage1_report["presampled_count"]
    relevant = stage1_report["survivor_count"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Collected (raw)", collected)
    c2.metric("Presampled / filtered", presampled)
    c3.metric("Relevant (Stage 1 output)", relevant)

    by_source = stage1_report.get("by_source", {})
    rows = [
        {"source": source, "survived": v["survived"], "of": v["of"], "rate": f"{v['rate']:.1%}"}
        for source, v in by_source.items()
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"{collected} snippets scraped -> presampled down to {presampled} before Stage 1 "
        f"(weighted by source, see ARCHITECTURE.md §2.3) -> {relevant} survived the relevance "
        f"filter and went into Stage 2 extraction."
    )


def run_live_pipeline(pasted_text: str) -> dict:
    lines = [line.strip() for line in pasted_text.splitlines() if line.strip()]
    snippets = [
        {"id": f"pasted_{i:04d}", "source": "pasted", "app": None, "date": None, "rating": None, "text": line}
        for i, line in enumerate(lines, start=1)
    ]

    survivors, _ = stage1_filter.classify_snippets(snippets, checkpoint_path=None)
    records, dropped = stage2_extract.extract_snippets(survivors)
    aggregates = stage3_aggregate.aggregate_records(records) if records else None

    return {
        "input_count": len(snippets),
        "survivor_count": len(survivors),
        "extracted_count": len(records),
        "dropped_count": len(dropped),
        "aggregates": aggregates,
        "records": records,
    }


st.title("AJIO Discovery Engine")
st.caption("Ranked, scored purchase-barrier analysis from public user feedback - not a summary.")

tab1, tab2, tab3, tab4 = st.tabs(["Run", "Opportunity table", "Evidence", "Questions"])

with tab1:
    st.subheader("Cached corpus results")
    st.write(
        "Loads the pre-computed pipeline output for the bundled corpus - no live model calls, "
        "renders instantly."
    )
    if st.button("Load cached results", type="primary"):
        cached = load_cached()
        st.session_state["cached"] = cached

    if "cached" in st.session_state:
        cached = st.session_state["cached"]
        if cached["aggregates"]:
            n = cached["aggregates"]["total_records"]
            st.success(f"Loaded - {n} classified snippets from the bundled corpus.")
            render_opportunity_table(cached["aggregates"], key_prefix="cached")
        else:
            st.warning("No cached aggregates found - run the pipeline first.")

    st.divider()
    st.subheader("Try it yourself")
    st.write(
        "Paste your own reviews below (one per line) and run Stages 1-3 live against them. "
        "This does not touch the cached corpus results above."
    )
    pasted = st.text_area(
        "One review per line",
        height=150,
        placeholder=(
            "Nice fabric but not sure about the size\n"
            "Saved this for a wedding, need to check if it's formal enough\n"
            "Too many similar kurtas in my wishlist, can't decide"
        ),
    )
    if st.button("Analyze my reviews"):
        if not pasted.strip():
            st.warning("Paste at least one review first.")
        else:
            with st.spinner("Running Stages 1-3 live..."):
                result = run_live_pipeline(pasted)
            st.success(
                f"{result['input_count']} pasted -> {result['survivor_count']} relevant -> "
                f"{result['extracted_count']} extracted ({result['dropped_count']} dropped)"
            )
            if result["aggregates"]:
                render_opportunity_table(result["aggregates"], key_prefix="live")
            else:
                st.info("None of the pasted reviews were classified as relevant to wishlisting/deferral.")

with tab2:
    st.subheader("Corpus provenance")
    st.write(
        "Sample size and where it came from — visible up front, since presenting the ranking "
        "without this context would make it easy to over-read a small, filtered sample."
    )
    cached = load_cached()
    render_corpus_provenance(cached["stage1_report"])

    st.divider()
    st.subheader("Opportunity table")
    render_opportunity_table(cached["aggregates"])

with tab3:
    st.subheader("Evidence")
    cached = load_cached()
    extractions = cached["extractions"]
    if not extractions:
        st.info("No extraction data available.")
    else:
        barriers = sorted({r["barrier"] for r in extractions})
        selected = st.selectbox("Barrier", barriers)
        rows = [r for r in extractions if r["barrier"] == selected]
        df = pd.DataFrame(rows)[["source", "evidence", "workaround", "intensity", "addressable_without_money"]]
        st.dataframe(df, width="stretch", hide_index=True)

with tab4:
    st.subheader("The ten questions")
    cached = load_cached()
    extractions = cached["extractions"] or []
    aggregates = cached["aggregates"] or {}
    barrier_table = aggregates.get("barrier_table", [])
    synthesis = cached["synthesis"] or {}

    def barrier_row(name):
        return next((b for b in barrier_table if b["barrier"] == name), None)

    save_intent_counts = Counter(r["save_intent"] for r in extractions)
    journey_saved_revisit = [r for r in extractions if r["journey_stage"] in ("saved", "revisit")]
    journey_barrier_counts = Counter(r["barrier"] for r in journey_saved_revisit)
    info_sought = [r["info_sought_outside_app"] for r in extractions if r.get("info_sought_outside_app")]
    workarounds = [r["workaround"] for r in extractions if r.get("workaround")]
    choice_overload_rows = [r for r in extractions if r["barrier"] == "choice_overload"]

    st.markdown("**1. Why do users add fashion products to their wishlist?**")
    st.write(f"`save_intent` distribution across {len(extractions)} classified snippets:")
    st.write(dict(save_intent_counts))

    st.markdown("**2. What prevents wishlisted products from eventually being purchased?**")
    st.write("Barrier ranking by Opportunity Score (see Tab 2 for the full table):")
    st.write([(b["barrier"], b["opportunity_score"]) for b in barrier_table[:3]])

    st.markdown("**3. What uncertainties remain after users identify a product they like?**")
    st.write(
        f"Barrier counts restricted to `journey_stage` in (saved, revisit) - "
        f"{len(journey_saved_revisit)} snippets:"
    )
    st.write(dict(journey_barrier_counts))

    st.markdown("**4. What causes users to postpone a purchase?**")
    st.write("Barriers ranked by mean intensity:")
    st.write(
        [(b["barrier"], b["mean_intensity"]) for b in sorted(barrier_table, key=lambda b: -b["mean_intensity"])[:3]]
    )

    st.markdown("**5. How do users compare multiple shortlisted products?**")
    co = barrier_row("choice_overload")
    st.write(f"`choice_overload`: {co['count'] if co else 0} snippets, example workarounds:")
    st.write([r["workaround"] for r in choice_overload_rows if r.get("workaround")][:5])

    st.markdown("**6. What information do users seek outside AJIO before purchasing?**")
    st.write(f"{len(info_sought)} snippets named something sought outside the app, e.g.:")
    st.write(info_sought[:5])

    st.markdown("**7. What role do fit, size, styling, price, reviews, occasion, social validation play?**")
    st.write("Full barrier table (see Tab 2).")
    st.write([(b["barrier"], b["count"], b["frequency_percent"]) for b in barrier_table])

    st.markdown("**8. When is the wishlist genuine purchase intent vs simply a bookmark?**")
    st.write("Barrier x save_intent cross-cut:")
    st.write(aggregates.get("barrier_x_save_intent", {}))

    st.markdown("**9. How do these behaviours differ across user segments?**")
    st.write("Barrier x segment_signal cross-cut:")
    st.write(aggregates.get("barrier_x_segment_signal", {}))

    st.markdown("**10. What unmet needs emerge consistently across user conversations?**")
    st.write(f"{len(workarounds)} workarounds observed, e.g.:")
    st.write(workarounds[:5])
    if synthesis.get("available"):
        st.write("Stage 4 synthesis:")
        st.markdown(synthesis["text"])
    else:
        st.info("Stage 4 synthesis unavailable for this run.")
