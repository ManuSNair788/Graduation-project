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

REDDIT_SUBREDDITS = ["IndianFashionAddicts", "india", "TwoXIndia", "OnexIndian"]
REDDIT_SEARCH_TERMS = ["wishlist", "AJIO", "Myntra", "online shopping size", "fashion returns India"]

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
# Reddit — PRAW, OAuth app-only (read-only) flow
#
# The public .json search endpoints started rejecting all unauthenticated requests
# (403/404 regardless of User-Agent — see git history for the earlier attempt). Switched to
# a registered "script" app's client id/secret in read-only mode, which needs no user login,
# just app credentials from reddit.com/prefs/apps. If REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET
# aren't set, this is logged and skipped per CONTEXT.md §9.7 rather than blocking the run.
# ---------------------------------------------------------------------------


def scrape_reddit(target_count: int) -> list[dict]:
    snippets: list[dict] = []
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "ajio-wishlist-research/1.0")

    if not client_id or not client_secret:
        log.error(
            "Reddit: REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not set, skipping "
            "(register a script app at reddit.com/prefs/apps — see .env.example)"
        )
        return snippets

    try:
        import praw

        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        reddit.read_only = True
    except Exception as exc:
        log.error(f"Reddit: PRAW auth setup FAILED, skipping: {exc}")
        return snippets

    per_combo = max(target_count // (len(REDDIT_SUBREDDITS) * len(REDDIT_SEARCH_TERMS)), 3)

    for sub_name in REDDIT_SUBREDDITS:
        if len(snippets) >= target_count:
            break
        try:
            subreddit = reddit.subreddit(sub_name)
        except Exception as exc:
            log.error(f"Reddit r/{sub_name} FAILED to open, skipping: {exc}")
            continue

        for term in REDDIT_SEARCH_TERMS:
            if len(snippets) >= target_count:
                break
            try:
                for submission in subreddit.search(term, limit=per_combo, sort="relevance"):
                    text = (submission.selftext or submission.title or "").strip()
                    if text:
                        snippets.append(
                            {
                                "id": _make_id("reddit"),
                                "source": "reddit",
                                "date": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d"),
                                "rating": None,
                                "text": text,
                            }
                        )

                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:5]:
                        c_text = (comment.body or "").strip()
                        if c_text and c_text not in ("[deleted]", "[removed]"):
                            snippets.append(
                                {
                                    "id": _make_id("reddit"),
                                    "source": "reddit",
                                    "date": datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d"),
                                    "rating": None,
                                    "text": c_text,
                                }
                            )
                    if len(snippets) >= target_count:
                        break
            except Exception as exc:
                log.error(f"Reddit r/{sub_name} search '{term}' FAILED, skipping: {exc}")

    log.info(
        f"Reddit: collected {len(snippets)} snippets across {len(REDDIT_SUBREDDITS)} subreddits "
        f"× {len(REDDIT_SEARCH_TERMS)} search terms"
    )
    return snippets


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
    all_snippets += scrape_reddit(target_count=300)
    all_snippets += scrape_youtube()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_snippets, f, ensure_ascii=False, indent=2)

    log.info(f"Wrote {len(all_snippets)} total raw snippets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
