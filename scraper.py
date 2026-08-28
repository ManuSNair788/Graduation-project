import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from google_play_scraper import Sort as GPSort
from google_play_scraper import reviews as gp_reviews

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scraper")

OUTPUT_PATH = Path("data/raw_snippets.json")

REDDIT_SEARCH_TERMS = ["AJIO", "Myntra wishlist", "online shopping size India", "fashion returns India"]
YOUTUBE_SEARCH_QUERIES = ["AJIO haul", "Myntra haul", "AJIO try-on review", "Myntra try-on review"]

_next_id = 0


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


def _make_id(source: str) -> str:
    global _next_id
    _next_id += 1
    return f"{source}_{_next_id:05d}"


# ---------------------------------------------------------------------------
# Google Play — AJIO (com.ril.ajio) + Myntra (com.myntra.android), for contrast
# ---------------------------------------------------------------------------


def scrape_google_play(app_id: str, label: str, target_count: int) -> list[dict]:
    snippets: list[dict] = []
    try:
        result, _ = gp_reviews(app_id, lang="en", country="in", sort=GPSort.NEWEST, count=target_count)
        for r in result:
            text = (r.get("content") or "").strip()
            if not text:
                continue
            at = r.get("at")
            snippets.append(
                {
                    "id": _make_id("play_store"),
                    "source": "play_store",
                    "date": at.strftime("%Y-%m-%d") if isinstance(at, datetime) else None,
                    "rating": r.get("score"),
                    "text": text,
                }
            )
        log.info(f"Google Play [{label}]: collected {len(snippets)} reviews")
    except Exception as exc:
        log.error(f"Google Play [{label}] FAILED, skipping: {exc}")
    return snippets


# ---------------------------------------------------------------------------
# App Store — AJIO (app id 1113425372)
#
# Known unreliable: the app-store-scraper library (unmaintained since ~2021) scrapes an
# undocumented Apple review-feed endpoint that returns non-JSON as of this run, per
# CONTEXT.md §9.7 ("scrapers depend on undocumented endpoints and change without notice").
# Explicitly the lowest-priority source ("whatever is available") — logged and skipped
# rather than debugged past the 20-minute cap.
# ---------------------------------------------------------------------------


def scrape_app_store(app_id: int, app_name: str, label: str, target_count: int) -> list[dict]:
    snippets: list[dict] = []
    try:
        from app_store_scraper import AppStore

        app = AppStore(country="in", app_name=app_name, app_id=app_id)
        app.review(how_many=target_count)
        for r in app.reviews:
            text = (r.get("review") or "").strip()
            if not text:
                continue
            date = r.get("date")
            snippets.append(
                {
                    "id": _make_id("app_store"),
                    "source": "app_store",
                    "date": date.strftime("%Y-%m-%d") if isinstance(date, datetime) else None,
                    "rating": r.get("rating"),
                    "text": text,
                }
            )
        log.info(f"App Store [{label}]: collected {len(snippets)} reviews")
    except Exception as exc:
        log.error(f"App Store [{label}] FAILED, skipping (known issue, see module docstring above): {exc}")
    return snippets


# ---------------------------------------------------------------------------
# Reddit — public .json search endpoints
#
# Known unreliable: as of this run, both www.reddit.com/search.json and
# old.reddit.com/search.json return 403/404 for unauthenticated requests regardless of
# User-Agent — Reddit appears to now block this outright. PRAW would need registered app
# credentials (CONTEXT.md §9.4) that aren't set up. Logged and skipped per §9.7; paste
# comments in manually if Reddit coverage is wanted.
# ---------------------------------------------------------------------------


def scrape_reddit(target_count: int) -> list[dict]:
    snippets: list[dict] = []
    user_agent = os.environ.get("REDDIT_USER_AGENT", "ajio-wishlist-research/1.0")
    headers = {"User-Agent": user_agent}
    per_term = max(target_count // len(REDDIT_SEARCH_TERMS), 1)

    for term in REDDIT_SEARCH_TERMS:
        try:
            r = requests.get(
                "https://www.reddit.com/search.json",
                params={"q": term, "limit": per_term, "sort": "relevance"},
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()
            posts = r.json().get("data", {}).get("children", [])
            for p in posts:
                d = p.get("data", {})
                text = (d.get("selftext") or d.get("title") or "").strip()
                if not text:
                    continue
                created = d.get("created_utc")
                snippets.append(
                    {
                        "id": _make_id("reddit"),
                        "source": "reddit",
                        "date": datetime.utcfromtimestamp(created).strftime("%Y-%m-%d") if created else None,
                        "rating": None,
                        "text": text,
                    }
                )
            time.sleep(2)  # courtesy delay between search terms
        except Exception as exc:
            log.error(f"Reddit search '{term}' FAILED, skipping: {exc}")

    log.info(f"Reddit: collected {len(snippets)} posts across {len(REDDIT_SEARCH_TERMS)} search terms")
    if not snippets:
        log.warning("Reddit: 0 snippets collected, see module docstring above for why.")
    return snippets


# ---------------------------------------------------------------------------
# YouTube — Data API v3 (search.list + commentThreads.list)
# ---------------------------------------------------------------------------


def scrape_youtube(target_count: int) -> list[dict]:
    snippets: list[dict] = []
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        log.error("YouTube: YOUTUBE_API_KEY not set, skipping")
        return snippets

    videos_per_query = 5
    comments_per_video = max(target_count // (len(YOUTUBE_SEARCH_QUERIES) * videos_per_query), 5)

    for query in YOUTUBE_SEARCH_QUERIES:
        try:
            search_r = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": videos_per_query,
                    "key": api_key,
                },
                timeout=15,
            )
            search_r.raise_for_status()
            video_ids = [item["id"]["videoId"] for item in search_r.json().get("items", [])]
        except Exception as exc:
            log.error(f"YouTube search '{query}' FAILED, skipping: {exc}")
            continue

        for video_id in video_ids:
            try:
                c_r = requests.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    params={
                        "part": "snippet",
                        "videoId": video_id,
                        "maxResults": comments_per_video,
                        "order": "relevance",
                        "key": api_key,
                    },
                    timeout=15,
                )
                if c_r.status_code == 403:
                    continue  # comments disabled on this video — not a scraper failure
                c_r.raise_for_status()
                for item in c_r.json().get("items", []):
                    top = item["snippet"]["topLevelComment"]["snippet"]
                    text = (top.get("textOriginal") or "").strip()
                    if not text:
                        continue
                    published = top.get("publishedAt")
                    snippets.append(
                        {
                            "id": _make_id("youtube"),
                            "source": "youtube",
                            "date": published[:10] if published else None,
                            "rating": None,
                            "text": text,
                        }
                    )
            except Exception as exc:
                log.error(f"YouTube comments for video {video_id} FAILED, skipping: {exc}")

    log.info(f"YouTube: collected {len(snippets)} comments across {len(YOUTUBE_SEARCH_QUERIES)} search queries")
    return snippets


def main() -> None:
    all_snippets: list[dict] = []

    all_snippets += scrape_google_play("com.ril.ajio", "AJIO", target_count=2000)
    all_snippets += scrape_google_play("com.myntra.android", "Myntra", target_count=1000)
    all_snippets += scrape_app_store(1113425372, "ajio", "AJIO", target_count=500)
    all_snippets += scrape_reddit(target_count=300)
    all_snippets += scrape_youtube(target_count=300)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_snippets, f, ensure_ascii=False, indent=2)

    log.info(f"Wrote {len(all_snippets)} total raw snippets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
