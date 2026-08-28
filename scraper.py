import json
import logging
import os
from datetime import datetime
from pathlib import Path

import requests
from google_play_scraper import Sort as GPSort
from google_play_scraper import reviews as gp_reviews

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scraper")

OUTPUT_PATH = Path("data/raw_snippets.json")

YOUTUBE_SEARCH_QUERIES = [
    "AJIO haul",
    "Myntra haul",
    "AJIO try-on review",
    "Myntra try-on review",
    "AJIO try on haul",
    "Myntra size guide",
    "online shopping fails India",
    "what I ordered vs what I got",
    "AJIO honest review",
    "Myntra haul review",
    "AJIO vs Myntra quality",
]
YOUTUBE_MAX_VIDEOS = 15
YOUTUBE_COMMENTS_PER_VIDEO = 100  # single page per video, API max

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
                    "app": label.lower(),
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
                    "app": label.lower(),
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
# Reddit — permanently dead, not attempted.
#
# Two routes tried and abandoned: (1) the public .json search endpoints, which now reject
# all unauthenticated requests (403/404 regardless of User-Agent); (2) PRAW OAuth app-only
# mode, which needs a "script" app registered at reddit.com/prefs/apps — that registration's
# captcha could not be completed, so no client id/secret exist and none will be pursued.
# Logged and skipped per CONTEXT.md §9.7, same treatment as App Store above. Left in as a
# stub (rather than deleted outright) so the source list in CONTEXT.md/ARCHITECTURE.md and
# the corpus's source coverage stay honestly documented in one place.
# ---------------------------------------------------------------------------


def scrape_reddit() -> list[dict]:
    log.warning("Reddit: permanently dead source, not attempted (see module docstring above)")
    return []


# ---------------------------------------------------------------------------
# YouTube — Data API v3 (search.list + commentThreads.list)
# ---------------------------------------------------------------------------


def scrape_youtube() -> list[dict]:
    snippets: list[dict] = []
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        log.error("YouTube: YOUTUBE_API_KEY not set, skipping")
        return snippets

    seen_video_ids: set[str] = set()
    video_queue: list[str] = []

    for query in YOUTUBE_SEARCH_QUERIES:
        if len(video_queue) >= YOUTUBE_MAX_VIDEOS:
            break
        try:
            search_r = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": 5,
                    "key": api_key,
                },
                timeout=15,
            )
            search_r.raise_for_status()
            for item in search_r.json().get("items", []):
                vid = item["id"]["videoId"]
                if vid not in seen_video_ids and len(video_queue) < YOUTUBE_MAX_VIDEOS:
                    seen_video_ids.add(vid)
                    video_queue.append(vid)
        except Exception as exc:
            log.error(f"YouTube search '{query}' FAILED, skipping: {exc}")

    log.info(f"YouTube: {len(video_queue)} unique videos queued across {len(YOUTUBE_SEARCH_QUERIES)} search queries")

    for video_id in video_queue:
        try:
            c_r = requests.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params={
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": YOUTUBE_COMMENTS_PER_VIDEO,
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
                        "app": None,
                        "date": published[:10] if published else None,
                        "rating": None,
                        "text": text,
                    }
                )
        except Exception as exc:
            log.error(f"YouTube comments for video {video_id} FAILED, skipping: {exc}")

    log.info(f"YouTube: collected {len(snippets)} comments across {len(video_queue)} videos")
    return snippets


def main() -> None:
    all_snippets: list[dict] = []

    all_snippets += scrape_google_play("com.ril.ajio", "AJIO", target_count=2000)
    all_snippets += scrape_google_play("com.myntra.android", "Myntra", target_count=1000)
    all_snippets += scrape_app_store(1113425372, "ajio", "AJIO", target_count=500)
    all_snippets += scrape_reddit()
    all_snippets += scrape_youtube()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_snippets, f, ensure_ascii=False, indent=2)

    log.info(f"Wrote {len(all_snippets)} total raw snippets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
