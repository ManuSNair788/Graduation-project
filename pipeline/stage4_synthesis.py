import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage4")

EXPECTED_OUTPUT_TOKENS = 600


def _build_prompt(aggregates: dict) -> str:
    table_json = json.dumps(aggregates["barrier_table"], indent=2)
    return (
        "Below is an aggregate table of purchase barriers extracted from user feedback about a "
        "fashion e-commerce wishlist, ranked by Opportunity Score. Based only on this table:\n\n"
        f"{table_json}\n\n"
        "1. Name the top three opportunity areas.\n"
        "2. Briefly say what distinguishes each from the others.\n"
        "3. Say which user segment each affects most, if inferable from the data.\n\n"
        "Keep the response short and structured — a few sentences per opportunity area, not a "
        "rewrite of the table."
    )


def run(input_path: str = "data/aggregates.json", output_path: str = "data/synthesis.json") -> dict:
    with open(input_path, encoding="utf-8") as f:
        aggregates = json.load(f)

    prompt = _build_prompt(aggregates)
    result = llm.call_synthesis(prompt, expected_output_tokens=EXPECTED_OUTPUT_TOKENS)

    if result.ok:
        synthesis = {"available": True, "text": result.text}
        log.info("Stage4: synthesis written")
    else:
        # Degrade gracefully — Tabs 2-4 must still render from Stage 3 alone (ARCHITECTURE.md §2.2/§4.1).
        synthesis = {"available": False, "error": result.error}
        log.error(f"Stage4: synthesis call failed ({result.error}), writing unavailable state")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, ensure_ascii=False, indent=2)

    return synthesis


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
