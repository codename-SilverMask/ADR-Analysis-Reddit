#!/usr/bin/env python3
"""
reddit_scraper.py -- Unified Reddit ADR (Adverse Drug Reaction) Analysis Pipeline

Merges functionality from reddit_adr_analyzer.py and analyze_comments.py into one
optimised end-to-end script.  Scrapes Reddit (with optional Pushshift fallback for
historical coverage), detects ADR mentions, scores severity/sentiment, computes
comment-level statistics, and generates all visualisations.

Usage:
    python reddit_scraper.py --subreddits ADHD,ADHDmemes --years 15
"""

# ──────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import requests.auth
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("reddit_scraper")

# ──────────────────────────────────────────────────────────────────────
# CREDENTIALS  (env-var overrides supported)
# ──────────────────────────────────────────────────────────────────────
CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "XXXXXXXXXXXXXXX")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "XXXXXXXXXXXXXXX")
USERNAME      = os.getenv("REDDIT_USERNAME",       "XXXXXXXXXXXXXXX")
PASSWORD      = os.getenv("REDDIT_PASSWORD",       "XXXXXXXXXXXXXXX")
USER_AGENT    = "RedditADRAnalyzer/3.0 (by /u/XXXXXXXXXXXXXXX)"
# ──────────────────────────────────────────────────────────────────────
# DEFAULT SUBREDDITS
# ──────────────────────────────────────────────────────────────────────
DEFAULT_SUBREDDITS = [
    "ADHD", "ADHDmemes", "adhdwomen", "adhd_anxiety",
    "ADHD_partners", "ADHDUK", "VyvanseADHD", "AdhdRelationships",
    'drug', 'DrugAddiction', 'HowDrugsWork', 'pharms', 'drugaddicts'
]

# ──────────────────────────────────────────────────────────────────────
# MEDICATION LIST
# ──────────────────────────────────────────────────────────────────────
MEDICATIONS: Set[str] = {
    # Stimulants
    "adderall", "vyvanse", "ritalin", "concerta", "focalin", "dexedrine",
    "methylphenidate", "lisdexamfetamine", "dextroamphetamine", "amphetamine",
    "mydayis", "quillivant", "daytrana", "evekeo", "zenzedi", "dyanavel",
    # Non-stimulants
    "strattera", "atomoxetine", "intuniv", "guanfacine", "kapvay", "clonidine",
    "wellbutrin", "bupropion", "qelbree", "viloxazine", "wellbutrin xl",
    # Antidepressants commonly used with ADHD
    "prozac", "fluoxetine", "zoloft", "sertraline", "lexapro", "escitalopram",
    "celexa", "citalopram", "effexor", "venlafaxine", "cymbalta", "duloxetine",
    "paxil", "paroxetine", "pristiq", "desvenlafaxine", "remeron", "mirtazapine",
    "trazodone", "trintellix", "vortioxetine", "celex", "effexor xr", "paxil cr",
    "viibryd", "vilazodone", "amitriptyline", "tramadol", "nefazodone", "nortriptyline",
    "desipramine", "doxepin", "amoxapine", "fluvoxamine", "imipramine", "ketamine",
    "nardil", "norpramin", "parnate", "phenelzine", "symbyax", "tranylcypromine",
    "clomipramine", "emsam", "pamelor", "selegiline", "esketamine", "isocarboxazid",
    "levomilnacipran", "marplan", "protriptyline", "spravato", "trimipramine",
    # Anxiety medications
    "xanax", "alprazolam", "ativan", "lorazepam", "klonopin", "clonazepam",
    "valium", "diazepam", "buspar", "buspirone", "hydroxyzine", "vistaril",
    "propranolol", "gabapentin", "neurontin", "lyrica", "pregabalin", "atenolol", "nadolol",
    "alprazolam intensol", "tranxene", "lorazepam intensol", "clorazepate",
    "diazepam intensol", "loreev xr", "oxazepam", "cannabidiol", "chlordiazepoxide",
    "meprobamate", "tranxene t-tab", "trifluoperazine", "oxcarbazepine", "phenytoin",
    # Mood stabilizers / Antipsychotics
    "abilify", "aripiprazole", "risperdal", "risperidone", "seroquel", "quetiapine",
    "zyprexa", "olanzapine", "latuda", "lurasidone", "rexulti", "brexpiprazole",
    "lamictal", "lamotrigine", "lithium", "depakote", "valproate", "paliperidone",
    # Supplements/Other
    "modafinil", "provigil", "armodafinil", "nuvigil", "amantadine", "deplin",
    "l-methylfolate", "methylin", "niacin", "methyl folate forte",
    "forfivo", "budeprion", "raldesy", "aplenzin", "fetzima", "irenka", "xaquil",
    # Slang terms
    "addy", "addies", "study meds", "focus pills", "concentration vitamins",
    "executive function pills", "brain fuel", "productivity pills", "smarties",
    "attention juice", "zoomies in a pill", "brain batteries",
    "antideps", "happy pills", "miracle drugs", "brain meds", "mood meds",
    "serotonin pills", "stability pills", "emotional support meds", "chemical helpers",
    "daily dose", "brain candy", "psych meds", "anti-depressy", "depression pills",
    "my silly little meds", "my scripts",
    # Psilocybin
    "boom", "boomers", "magic mushrooms", "simple simon", "caps", "shrooms",
    "buttons", "mushies", "psilocybin",
    # LSD
    "lsd", "haze", "california sunshine", "white lightning", "mellow yellow",
    "tabs", "trips", "microdots", "microdose", "dots", "blotter", "cubes",
    # Cannabis
    "weed", "za", "sticky", "grass", "exotic", "flower", "gas", "wax pen",
    "oil", "dab", "amnesia", "bizarro", "blaze", "bliss", "brain freeze",
    "buzz haze", "cloud 10", "dr. feel good", "green peace", "geeked",
    "geeked up", "sky high", "snake bite", "potpourri", "herbal incense",
    "blunt", "bowl", "bong", "bubbler", "doobie", "fatty", "gravity bong",
    "j", "jay", "joint", "left-handed cigarette", "one-hitter", "percolator",
    "piece", "pipe", "rig", "roach", "spliff", "vape", "water pipe", "cannabis",
    "marijuana", "thc", "cbd", "edibles",
}

# Pre-compile medication patterns for speed
_MED_PATTERNS: Dict[str, re.Pattern] = {
    med: re.compile(r"\b" + re.escape(med) + r"\b", re.IGNORECASE)
    for med in MEDICATIONS
}

# ──────────────────────────────────────────────────────────────────────
# ADR KEYWORDS (by category)
# ──────────────────────────────────────────────────────────────────────
ADR_KEYWORDS: Dict[str, List[str]] = {
    "side_effect_indicators": [
        "side effect", "side-effect", "adverse", "reaction", "bad reaction",
        "caused", "making me", "made me", "giving me", "experiencing",
        "suffering from", "dealing with", "struggling with", "issues with",
        "problems with", "concern", "worried about", "scared of",
        "negative effect", "downsides", "drawbacks", "complications",
        "affecting me", "messing me up", "screwing with", "not right",
        "feeling off", "something wrong", "not myself", "weird symptoms",
    ],
    "withdrawal_tolerance": [
        "withdrawal", "withdrawing", "coming off", "stopping", "quit",
        "discontinuation", "tolerance", "not working anymore", "stopped working",
        "losing effect", "building tolerance", "wears off", "crash", "rebound",
        "tapering", "weaning off", "cold turkey", "detox", "dependency",
        "addicted", "addiction", "dependence", "need more", "need higher dose",
        "doesnt work", "doesn't work", "lost effectiveness", "reduced effect",
    ],
    "physical_symptoms": [
        "headache", "migraine", "nausea", "vomiting", "diarrhea", "constipation",
        "stomach pain", "appetite loss", "weight loss", "weight gain",
        "dry mouth", "sweating", "tremor", "shaking", "dizziness", "dizzy",
        "fatigue", "tired", "exhausted", "insomnia", "cant sleep", "can't sleep",
        "sleep issues", "racing heart", "palpitations", "chest pain",
        "shortness of breath", "tics", "twitching", "muscle pain", "joint pain",
        "rash", "itching", "blurred vision", "tinnitus", "ringing ears", "numbness",
        "abdominal pain", "bloating", "gas", "indigestion", "acid reflux",
        "hot flashes", "cold sweats", "night sweats", "chills", "fever",
        "weakness", "lethargy", "sluggish", "heavy limbs", "body aches",
        "restless legs", "jaw clenching", "teeth grinding", "bruxism",
        "hair loss", "skin problems", "acne", "hives", "swelling",
        "back pain", "stiff neck", "sore muscles", "cramping", "spasms",
        "vertigo", "lightheaded", "unsteady", "balance issues",
    ],
    "psychological_symptoms": [
        "anxiety", "anxious", "panic", "depression", "depressed", "sad",
        "irritable", "irritability", "angry", "rage", "mood swings", "emotional",
        "crying", "suicidal", "suicide", "self harm", "intrusive thoughts",
        "paranoid", "paranoia", "hallucination", "zombie", "flat", "numb",
        "brain fog", "foggy", "confused", "memory loss", "forgetful",
        "dissociation", "derealisation", "depersonalization",
        "panic attacks", "anxiety attacks", "agitation", "restless", "on edge",
        "nightmares", "vivid dreams", "sleep paralysis", "night terrors",
        "hopeless", "worthless", "guilt", "shame", "emptiness",
        "anhedonia", "no motivation", "apathy", "emotionally blunted",
        "manic", "hypomania", "racing thoughts", "impulsive", "reckless",
        "obsessive", "compulsive", "rumination", "overthinking",
        "derealization", "detached", "unreal", "spacey", "out of it",
    ],
    "cardiovascular": [
        "heart rate", "blood pressure", "hypertension", "tachycardia",
        "arrhythmia", "chest tightness", "fainting", "syncope",
        "elevated heart rate", "high blood pressure", "low blood pressure",
        "irregular heartbeat", "skipped beats", "heart pounding",
    ],
    "cognitive": [
        "brain zaps", "zaps", "focus worse", "cant focus", "can't focus",
        "concentration", "attention span", "memory", "cognitive",
        "mental fog", "thinking slower", "processing slow", "word finding",
        "cant think", "can't think", "confusion", "disoriented",
        "short term memory", "forgetting things", "lost my train of thought",
    ],
    "gastrointestinal": [
        "stomach issues", "digestive problems", "upset stomach", "queasy",
        "gi issues", "bowel problems", "ibs symptoms",
    ],
    "neurological": [
        "seizures", "convulsions", "nerve pain", "neuropathy",
        "pins and needles", "tingling", "burning sensation", "electric shocks",
    ],
}

SEVERITY_INDICATORS: Dict[str, List[str]] = {
    "severe": [
        "severe", "extreme", "unbearable", "terrible", "awful", "horrible",
        "emergency", "hospitalized", "er visit", "urgent care",
        "doctor immediately", "cant function", "can't function", "unable to",
        "debilitating", "excruciating", "intolerable", "life threatening",
        "life-threatening", "crisis", "critical", "dangerous", "scary",
        "terrifying", "worst ever", "worst pain", "never felt this bad",
        "thought i was dying", "called 911", "ambulance", "suicide watch",
        "psychiatric hold", "cant work", "can't work", "cant leave bed",
        "can't leave bed", "bedridden", "incapacitated",
        "completely non-functional",
    ],
    "moderate": [
        "bad", "worse", "concerning", "worrying", "uncomfortable", "difficult",
        "hard to", "struggling", "noticeable", "significant", "pretty bad",
        "rough", "tough", "challenging", "problematic", "interfering",
        "affecting daily life", "impacting work", "cant ignore",
        "can't ignore", "distracting", "bothersome", "unpleasant",
        "miserable", "suffering", "hard to deal with",
    ],
    "mild": [
        "mild", "slight", "minor", "a little", "somewhat", "manageable",
        "tolerable", "annoying", "inconvenient", "barely noticeable", "hardly",
        "not too bad", "livable", "can deal with", "can handle", "bearable",
        "acceptable", "minor nuisance", "small issue", "tiny bit", "occasionally",
    ],
}

POSITIVE_INDICATORS: List[str] = [
    "helped", "helping", "better", "improved", "improvement", "great", "amazing",
    "wonderful", "working well", "life changing", "life-changing", "best decision",
    "no side effects", "no issues", "minimal side effects", "worth it",
    "game changer", "miracle", "blessed", "grateful", "thankful",
    "finally working", "doing great", "feel good", "feel better", "feeling better",
    "highly recommend", "recommend", "love it", "love this", "works great",
    "works well", "effective", "positive effects", "positive experience",
    "no complaints", "happy with", "satisfied", "perfect", "excellent",
    "outstanding", "fantastic", "incredible", "best thing", "saved my life",
    "totally fine", "all good", "going well", "success", "successful",
    "beneficial", "advantageous", "helpful", "productive", "focus improved",
    "mood improved", "anxiety reduced", "depression lifted", "symptom free",
]

# ──────────────────────────────────────────────────────────────────────
# TUNABLES
# ──────────────────────────────────────────────────────────────────────
POSTS_PER_PAGE         = 100
MAX_LISTING_PAGES      = 10     # max pages per listing endpoint call
RATE_LIMIT_SLEEP       = 2.0   # seconds between paginated listing requests
COMMENT_FETCH_SLEEP    = 1.0   # seconds between individual-post comment fetches
PUSHSHIFT_PAGE_SIZE    = 100
PUSHSHIFT_SLEEP        = 1.0
SEARCH_SLEEP           = 2.0

# ──────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ──────────────────────────────────────────────────────────────────────

def authenticate() -> Tuple[bool, dict, str]:
    """
    Attempt Reddit OAuth authentication.
    Returns (use_oauth, headers, base_endpoint).
    """
    try:
        client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
        post_data = {
            "grant_type": "password",
            "username": USERNAME,
            "password": PASSWORD,
        }
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            data=post_data,
            headers={"User-Agent": USER_AGENT},
            auth=client_auth,
            timeout=15,
        )
        if resp.status_code == 200:
            token = resp.json().get("access_token")
            if token:
                log.info("OAuth authentication successful")
                return (
                    True,
                    {"User-Agent": USER_AGENT, "Authorization": f"bearer {token}"},
                    "https://oauth.reddit.com",
                )
        log.warning("OAuth token not obtained (status %s); falling back to public API", resp.status_code)
    except Exception as exc:
        log.warning("OAuth authentication failed (%s); falling back to public API", exc)

    return (
        False,
        {"User-Agent": USER_AGENT},
        "https://www.reddit.com",
    )


# ──────────────────────────────────────────────────────────────────────
# ENGAGEMENT HELPERS
# ──────────────────────────────────────────────────────────────────────

def _epoch_to_human(epoch: Optional[float]) -> Optional[str]:
    """Convert epoch seconds to ISO-format string."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OSError, ValueError):
        return None


def _estimate_votes(score: int, upvote_ratio: Optional[float]) -> Tuple[int, int]:
    """
    Estimate upvotes / downvotes from score + upvote_ratio.
    Reddit fuzzes these numbers; this is an approximation.
    """
    if upvote_ratio is None or upvote_ratio <= 0 or upvote_ratio >= 1:
        # Can't estimate -- return score as upvotes, 0 downvotes
        return (max(score, 0), max(-score, 0))
    denom = 2.0 * upvote_ratio - 1.0
    if abs(denom) < 1e-6:
        return (max(score, 0), max(-score, 0))
    total = score / denom
    upvotes = round(total * upvote_ratio)
    downvotes = round(total * (1.0 - upvote_ratio))
    return (max(upvotes, 0), max(downvotes, 0))


def _compute_engagement_ratio(score: int, num_comments: int, created_utc: Optional[float]) -> Optional[float]:
    """engagement_ratio = (score + num_comments) / post_age_hours."""
    if created_utc is None:
        return None
    age_hours = (time.time() - created_utc) / 3600.0
    if age_hours <= 0:
        return None
    return (score + num_comments) / age_hours


# ──────────────────────────────────────────────────────────────────────
# DATA COLLECTION -- REDDIT API
# ──────────────────────────────────────────────────────────────────────

def _safe_request(url: str, headers: dict, params: dict, timeout: int = 20) -> Optional[requests.Response]:
    """Wrapper with retry + back-off."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                log.warning("Rate-limited; sleeping %ds", wait)
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            log.warning("Request failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return None


def fetch_listing_posts(
    subreddit: str,
    use_oauth: bool,
    headers: dict,
    base_endpoint: str,
    sort: str = "new",
    time_filter: str = "all",
    max_pages: int = MAX_LISTING_PAGES,
) -> List[Dict]:
    """
    Paginate through a subreddit listing endpoint (/new, /top, /hot, etc.).
    Returns list of post dicts with engagement metrics.
    """
    collected: List[Dict] = []
    after_key: Optional[str] = None
    seen_ids: Set[str] = set()

    for page in range(max_pages):
        params: Dict[str, Any] = {"limit": POSTS_PER_PAGE, "t": time_filter}
        if after_key:
            params["after"] = after_key

        suffix = "" if use_oauth else ".json"
        url = f"{base_endpoint}/r/{subreddit}/{sort}{suffix}"
        time.sleep(RATE_LIMIT_SLEEP)
        resp = _safe_request(url, headers, params)
        if resp is None or resp.status_code != 200:
            break

        try:
            data = resp.json()
        except ValueError:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            pd_ = child.get("data", {})
            pid = pd_.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            score = pd_.get("score", 0)
            upvote_ratio = pd_.get("upvote_ratio")
            num_comments = pd_.get("num_comments", 0)
            created_utc = pd_.get("created_utc")
            upvotes, downvotes = _estimate_votes(score, upvote_ratio)

            collected.append({
                "subreddit": subreddit,
                "id": pid,
                "title": pd_.get("title", ""),
                "selftext": pd_.get("selftext", ""),
                "author": pd_.get("author", "[deleted]"),
                "created_utc": created_utc,
                "posted_time": _epoch_to_human(created_utc),
                "score": score,
                "upvotes": upvotes,
                "downvotes": downvotes,
                "upvote_ratio": upvote_ratio,
                "num_comments": num_comments,
                "engagement_ratio": _compute_engagement_ratio(score, num_comments, created_utc),
                "url": pd_.get("url", ""),
            })

        after_key = data.get("data", {}).get("after")
        if not after_key:
            break
        log.info("  [%s/%s] page %d  -> %d posts so far", subreddit, sort, page + 1, len(collected))

    return collected


def fetch_search_posts(
    subreddit: str,
    query: str,
    use_oauth: bool,
    headers: dict,
    base_endpoint: str,
    sort: str = "new",
    time_filter: str = "all",
    max_pages: int = MAX_LISTING_PAGES,
) -> List[Dict]:
    """
    Use Reddit search endpoint to find posts matching *query* within a subreddit.
    Useful for surfacing older posts that listing endpoints cannot reach.
    """
    collected: List[Dict] = []
    after_key: Optional[str] = None
    seen_ids: Set[str] = set()

    for page in range(max_pages):
        params: Dict[str, Any] = {
            "q": query,
            "restrict_sr": "on",
            "sort": sort,
            "t": time_filter,
            "limit": POSTS_PER_PAGE,
            "type": "link",
        }
        if after_key:
            params["after"] = after_key

        suffix = "" if use_oauth else ".json"
        url = f"{base_endpoint}/r/{subreddit}/search{suffix}"
        time.sleep(SEARCH_SLEEP)
        resp = _safe_request(url, headers, params)
        if resp is None or resp.status_code != 200:
            break

        try:
            data = resp.json()
        except ValueError:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            pd_ = child.get("data", {})
            pid = pd_.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            score = pd_.get("score", 0)
            upvote_ratio = pd_.get("upvote_ratio")
            num_comments = pd_.get("num_comments", 0)
            created_utc = pd_.get("created_utc")
            upvotes, downvotes = _estimate_votes(score, upvote_ratio)

            collected.append({
                "subreddit": subreddit,
                "id": pid,
                "title": pd_.get("title", ""),
                "selftext": pd_.get("selftext", ""),
                "author": pd_.get("author", "[deleted]"),
                "created_utc": created_utc,
                "posted_time": _epoch_to_human(created_utc),
                "score": score,
                "upvotes": upvotes,
                "downvotes": downvotes,
                "upvote_ratio": upvote_ratio,
                "num_comments": num_comments,
                "engagement_ratio": _compute_engagement_ratio(score, num_comments, created_utc),
                "url": pd_.get("url", ""),
            })

        after_key = data.get("data", {}).get("after")
        if not after_key:
            break
        log.info("  [search/%s] page %d  -> %d posts", subreddit, page + 1, len(collected))

    return collected


def fetch_post_comments(
    subreddit: str,
    post_id: str,
    use_oauth: bool,
    base_endpoint: str,
    headers: dict,
) -> List[Dict]:
    """Recursively fetch all comments for a single post, including engagement data."""
    comments: List[Dict] = []
    suffix = "" if use_oauth else ".json"
    url = f"{base_endpoint}/r/{subreddit}/comments/{post_id}{suffix}"
    resp = _safe_request(url, headers, {})
    if resp is None or resp.status_code != 200:
        return comments

    try:
        data = resp.json()
    except ValueError:
        return comments

    if not isinstance(data, list) or len(data) < 2:
        return comments

    def _walk(children: list, depth: int = 0):
        for item in children:
            if item.get("kind") != "t1":
                continue
            cd = item.get("data", {})
            created_utc = cd.get("created_utc")
            score = cd.get("score", 0)
            comments.append({
                "post_id": post_id,
                "subreddit": subreddit,
                "comment_id": cd.get("id"),
                "body": cd.get("body", ""),
                "author": cd.get("author", "[deleted]"),
                "created_utc": created_utc,
                "posted_time": _epoch_to_human(created_utc),
                "score": score,
                "upvotes": max(score, 0),   # comments don't expose upvote_ratio
                "downvotes": max(-score, 0),
                "depth": depth,
            })
            replies = cd.get("replies")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                _walk(reply_children, depth + 1)

    _walk(data[1].get("data", {}).get("children", []))
    return comments


# ──────────────────────────────────────────────────────────────────────
# DATA COLLECTION -- PUSHSHIFT (historical)
# ──────────────────────────────────────────────────────────────────────
PUSHSHIFT_SUBMISSIONS_URL = "https://api.pushshift.io/reddit/search/submission"
PUSHSHIFT_COMMENTS_URL    = "https://api.pushshift.io/reddit/search/comment"


def _pushshift_available() -> bool:
    """Quick check whether Pushshift API responds."""
    try:
        r = requests.get(
            PUSHSHIFT_SUBMISSIONS_URL,
            params={"subreddit": "test", "size": 1},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def fetch_pushshift_posts(
    subreddit: str,
    after_epoch: int,
    before_epoch: int,
) -> List[Dict]:
    """
    Fetch historical posts from Pushshift for a given time window.
    Walks forward from *after_epoch* in pages of PUSHSHIFT_PAGE_SIZE.
    """
    collected: List[Dict] = []
    cursor_after = after_epoch

    while cursor_after < before_epoch:
        params = {
            "subreddit": subreddit,
            "after": cursor_after,
            "before": before_epoch,
            "size": PUSHSHIFT_PAGE_SIZE,
            "sort": "asc",
            "sort_type": "created_utc",
        }
        time.sleep(PUSHSHIFT_SLEEP)
        try:
            r = requests.get(PUSHSHIFT_SUBMISSIONS_URL, params=params, timeout=30)
            if r.status_code != 200:
                log.warning("Pushshift returned %d for r/%s", r.status_code, subreddit)
                break
            items = r.json().get("data", [])
        except Exception as exc:
            log.warning("Pushshift request error: %s", exc)
            break

        if not items:
            break

        for item in items:
            created_utc = item.get("created_utc")
            score = item.get("score", 0)
            num_comments = item.get("num_comments", 0)
            upvote_ratio = item.get("upvote_ratio")
            upvotes, downvotes = _estimate_votes(score, upvote_ratio)

            collected.append({
                "subreddit": subreddit,
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "selftext": item.get("selftext", ""),
                "author": item.get("author", "[deleted]"),
                "created_utc": created_utc,
                "posted_time": _epoch_to_human(created_utc),
                "score": score,
                "upvotes": upvotes,
                "downvotes": downvotes,
                "upvote_ratio": upvote_ratio,
                "num_comments": num_comments,
                "engagement_ratio": _compute_engagement_ratio(score, num_comments, created_utc),
                "url": item.get("url", ""),
            })

        # advance cursor past last item
        cursor_after = int(items[-1].get("created_utc", cursor_after)) + 1
        log.info("  [pushshift/%s] %d posts (cursor at %s)",
                 subreddit, len(collected),
                 _epoch_to_human(cursor_after))

    return collected


def fetch_pushshift_comments(
    subreddit: str,
    after_epoch: int,
    before_epoch: int,
) -> List[Dict]:
    """Fetch historical comments from Pushshift for a given time window."""
    collected: List[Dict] = []
    cursor_after = after_epoch

    while cursor_after < before_epoch:
        params = {
            "subreddit": subreddit,
            "after": cursor_after,
            "before": before_epoch,
            "size": PUSHSHIFT_PAGE_SIZE,
            "sort": "asc",
            "sort_type": "created_utc",
        }
        time.sleep(PUSHSHIFT_SLEEP)
        try:
            r = requests.get(PUSHSHIFT_COMMENTS_URL, params=params, timeout=30)
            if r.status_code != 200:
                break
            items = r.json().get("data", [])
        except Exception:
            break

        if not items:
            break

        for item in items:
            created_utc = item.get("created_utc")
            score = item.get("score", 0)
            collected.append({
                "post_id": item.get("link_id", "").replace("t3_", ""),
                "subreddit": subreddit,
                "comment_id": item.get("id", ""),
                "body": item.get("body", ""),
                "author": item.get("author", "[deleted]"),
                "created_utc": created_utc,
                "posted_time": _epoch_to_human(created_utc),
                "score": score,
                "upvotes": max(score, 0),
                "downvotes": max(-score, 0),
                "depth": 0,  # Pushshift doesn't expose depth easily
            })

        cursor_after = int(items[-1].get("created_utc", cursor_after)) + 1
        log.info("  [pushshift-comments/%s] %d comments", subreddit, len(collected))

    return collected


# ──────────────────────────────────────────────────────────────────────
# ORCHESTRATE COLLECTION
# ──────────────────────────────────────────────────────────────────────

# Medication search terms (subset of branded names for search queries)
_SEARCH_TERMS = [
    "adderall", "vyvanse", "ritalin", "concerta", "strattera", "wellbutrin",
    "prozac", "zoloft", "lexapro", "xanax", "seroquel", "lamictal",
    "modafinil", "gabapentin", "side effect", "adverse reaction", "withdrawal",
]


def collect_all_data(
    subreddits: List[str],
    years: int,
    use_oauth: bool,
    headers: dict,
    base_endpoint: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Master collection routine.

    Strategy (ordered by preference):
    1. Pushshift (if available) -- walks full time window.
    2. Reddit listing endpoints (/new, /top?t=all) -- limited to ~1000 per sort.
    3. Reddit search -- medication-keyword queries to surface older posts.

    Comments are fetched per-post via Reddit API in all cases.
    """
    now_epoch = int(time.time())
    start_epoch = now_epoch - (years * 365 * 24 * 3600)

    all_posts: List[Dict] = []
    all_comments: List[Dict] = []
    global_post_ids: Set[str] = set()

    pushshift_ok = _pushshift_available()
    if pushshift_ok:
        log.info("Pushshift API is reachable -- will use for historical data")
    else:
        log.info("Pushshift API unavailable -- using Reddit API strategies only")

    for sub in subreddits:
        log.info("=" * 50)
        log.info("Collecting r/%s  (target window: %d years)", sub, years)
        log.info("=" * 50)
        sub_post_ids: Set[str] = set()

        # ---- Strategy 1: Pushshift ----
        if pushshift_ok:
            log.info("[1/3] Pushshift historical posts for r/%s ...", sub)
            ps_posts = fetch_pushshift_posts(sub, start_epoch, now_epoch)
            for p in ps_posts:
                if p["id"] not in sub_post_ids:
                    sub_post_ids.add(p["id"])
                    all_posts.append(p)
            log.info("  Pushshift yielded %d unique posts", len(ps_posts))

            log.info("[1/3] Pushshift historical comments for r/%s ...", sub)
            ps_comments = fetch_pushshift_comments(sub, start_epoch, now_epoch)
            all_comments.extend(ps_comments)
            log.info("  Pushshift yielded %d comments", len(ps_comments))

        # ---- Strategy 2: Reddit listing endpoints ----
        for sort_mode in ("new", "top", "hot", "rising"):
            log.info("[2/3] Reddit /%s listing for r/%s ...", sort_mode, sub)
            listing_posts = fetch_listing_posts(
                sub, use_oauth, headers, base_endpoint,
                sort=sort_mode, time_filter="all",
            )
            added = 0
            for p in listing_posts:
                if p["id"] not in sub_post_ids:
                    sub_post_ids.add(p["id"])
                    all_posts.append(p)
                    added += 1
            log.info("  /%s added %d new posts (total for sub: %d)", sort_mode, added, len(sub_post_ids))

        # ---- Strategy 3: Search for medication keywords ----
        log.info("[3/3] Reddit search queries for r/%s ...", sub)
        for term in _SEARCH_TERMS:
            search_posts = fetch_search_posts(
                sub, term, use_oauth, headers, base_endpoint,
                sort="new", time_filter="all", max_pages=3,
            )
            added = 0
            for p in search_posts:
                if p["id"] not in sub_post_ids:
                    sub_post_ids.add(p["id"])
                    all_posts.append(p)
                    added += 1
            if added:
                log.info("  search '%s' added %d new posts", term, added)

        # ---- Fetch comments for all posts via Reddit API ----
        log.info("Fetching comments for %d posts in r/%s ...", len(sub_post_ids), sub)
        # Determine which posts still need comments fetched (all from Reddit API;
        # Pushshift comments may cover some but Reddit API gives engagement data)
        posts_needing_comments = [
            p for p in all_posts
            if p["id"] in sub_post_ids and p.get("num_comments", 0) > 0
        ]
        # To avoid re-fetching, track comment post_ids already covered
        existing_comment_pids = {c["post_id"] for c in all_comments}
        fetch_count = 0
        for p in posts_needing_comments:
            if p["id"] in existing_comment_pids:
                continue
            time.sleep(COMMENT_FETCH_SLEEP)
            cmts = fetch_post_comments(sub, p["id"], use_oauth, base_endpoint, headers)
            all_comments.extend(cmts)
            fetch_count += 1
            if fetch_count % 50 == 0:
                log.info("  ... fetched comments for %d / %d posts", fetch_count, len(posts_needing_comments))

        global_post_ids.update(sub_post_ids)
        log.info("r/%s done: %d posts, %d comments collected so far", sub, len(sub_post_ids), len(all_comments))

    log.info("=" * 60)
    log.info("COLLECTION COMPLETE: %d posts, %d comments across %d subreddits",
             len(all_posts), len(all_comments), len(subreddits))
    log.info("=" * 60)
    return all_posts, all_comments


# ──────────────────────────────────────────────────────────────────────
# ADR ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def find_medications(text: str) -> Set[str]:
    """Return set of medication names found in *text*."""
    found: Set[str] = set()
    for med, pat in _MED_PATTERNS.items():
        if pat.search(text):
            found.add(med)
    return found


def extract_adr_context(text: str, med_name: str, window: int = 200) -> List[Dict]:
    """Extract ADR-bearing context windows around each mention of *med_name*."""
    text_lower = text.lower()
    pattern = _MED_PATTERNS[med_name]
    contexts: List[Dict] = []

    for match in pattern.finditer(text):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        ctx = text[start:end]
        ctx_lower = ctx.lower()

        adr_types: List[str] = []
        symptoms: List[str] = []
        severity: Optional[str] = None
        is_positive = False

        for category, kws in ADR_KEYWORDS.items():
            for kw in kws:
                if kw in ctx_lower:
                    adr_types.append(category)
                    symptoms.append(kw)

        for sev_level, kws in SEVERITY_INDICATORS.items():
            for kw in kws:
                if kw in ctx_lower:
                    severity = sev_level
                    break
            if severity:
                break

        for pos in POSITIVE_INDICATORS:
            if pos in ctx_lower:
                is_positive = True
                break

        if adr_types and not is_positive:
            contexts.append({
                "medication": med_name,
                "context": ctx.strip(),
                "adr_types": list(set(adr_types)),
                "symptoms": list(set(symptoms)),
                "severity": severity,
            })

    return contexts


def calculate_sentiment_score(context: str) -> float:
    """Simple keyword-ratio sentiment: -1 (negative) .. +1 (positive)."""
    ctx_lower = context.lower()
    neg = sum(1 for kws in ADR_KEYWORDS.values() for kw in kws if kw in ctx_lower)
    neg += sum(1 for kws in SEVERITY_INDICATORS.values() for kw in kws if kw in ctx_lower)
    pos = sum(1 for kw in POSITIVE_INDICATORS if kw in ctx_lower)
    total = neg + pos
    if total == 0:
        return 0.0
    return (pos - neg) / total


def analyze_adr_mentions(posts: List[Dict], comments: List[Dict]) -> List[Dict]:
    """
    Scan every post and comment for ADR mentions.
    Returns list of ADR-mention records enriched with engagement data.
    """
    adr_data: List[Dict] = []

    log.info("Analyzing %d posts for ADR mentions ...", len(posts))
    for p in posts:
        text = f"{p['title']} {p['selftext']}"
        meds = find_medications(text)
        for med in meds:
            for ctx in extract_adr_context(text, med):
                adr_data.append({
                    "source": "post",
                    "source_id": p["id"],
                    "post_id": p["id"],
                    "comment_id": None,
                    "subreddit": p["subreddit"],
                    "medication": med,
                    "author": p["author"],
                    "created_utc": p["created_utc"],
                    "posted_time": p.get("posted_time"),
                    "timestamp": _epoch_to_human(p["created_utc"]),
                    "score": p["score"],
                    "net_score": p["score"],
                    "upvotes": p.get("upvotes", 0),
                    "downvotes": p.get("downvotes", 0),
                    "num_comments": p.get("num_comments", 0),
                    "engagement_ratio": p.get("engagement_ratio"),
                    "adr_types": ctx["adr_types"],
                    "symptoms": ctx["symptoms"],
                    "severity": ctx["severity"],
                    "context": ctx["context"],
                    "sentiment": calculate_sentiment_score(ctx["context"]),
                })

    log.info("Analyzing %d comments for ADR mentions ...", len(comments))
    for c in comments:
        meds = find_medications(c["body"])
        for med in meds:
            for ctx in extract_adr_context(c["body"], med):
                adr_data.append({
                    "source": "comment",
                    "source_id": c["comment_id"],
                    "post_id": c["post_id"],
                    "comment_id": c["comment_id"],
                    "subreddit": c.get("subreddit", ""),
                    "medication": med,
                    "author": c["author"],
                    "created_utc": c["created_utc"],
                    "posted_time": c.get("posted_time"),
                    "timestamp": _epoch_to_human(c["created_utc"]),
                    "score": c["score"],
                    "net_score": c["score"],
                    "upvotes": c.get("upvotes", 0),
                    "downvotes": c.get("downvotes", 0),
                    "num_comments": 0,
                    "engagement_ratio": None,
                    "adr_types": ctx["adr_types"],
                    "symptoms": ctx["symptoms"],
                    "severity": ctx["severity"],
                    "context": ctx["context"],
                    "sentiment": calculate_sentiment_score(ctx["context"]),
                })

    log.info("Total ADR mentions detected: %d", len(adr_data))
    return adr_data


# ──────────────────────────────────────────────────────────────────────
# COMMENT-LEVEL ADR STATISTICS  (from analyze_comments.py)
# ──────────────────────────────────────────────────────────────────────

def compute_comment_adr_statistics(
    posts_df: pd.DataFrame,
    adr_df: pd.DataFrame,
) -> None:
    """Print comprehensive comment-ADR stats (mirrors analyze_comments.py output)."""
    if adr_df.empty or posts_df.empty:
        log.info("No data for comment ADR statistics")
        return

    adr_comments_df = adr_df[adr_df["source"] == "comment"].copy()
    adr_posts_df = adr_df[adr_df["source"] == "post"]

    if adr_comments_df.empty:
        log.info("No comment-sourced ADR mentions found")
        return

    adr_per_post = adr_comments_df.groupby("post_id").size().reset_index(name="adr_comment_count")
    posts_lookup = posts_df.set_index("id")["subreddit"].to_dict()
    adr_per_post["subreddit"] = adr_per_post["post_id"].map(posts_lookup)
    adr_per_post.dropna(subset=["subreddit"], inplace=True)
    counts = adr_per_post["adr_comment_count"]

    print("\n" + "=" * 60)
    print("ADR COMMENTS PER POST STATISTICS")
    print("=" * 60)
    print(f"  Total Posts Analyzed:          {len(posts_df):,}")
    print(f"  Total ADR Mentions:            {len(adr_df):,}")
    print(f"    - From Posts:                {len(adr_posts_df):,}")
    print(f"    - From Comments:             {len(adr_comments_df):,}")
    print(f"  Posts w/ >=1 ADR comment:      {len(adr_per_post):,}")
    print()
    print("  --- Per-Post Statistics ---")
    print(f"    Min:    {counts.min()}")
    print(f"    Max:    {counts.max()}")
    print(f"    Mean:   {counts.mean():.2f}")
    print(f"    Median: {counts.median():.1f}")
    print(f"    Std:    {counts.std():.2f}")
    print()
    print("  --- Distribution ---")
    print(f"    1 ADR comment:      {(counts == 1).sum():,}")
    print(f"    2-5 ADR comments:   {((counts >= 2) & (counts <= 5)).sum():,}")
    print(f"    6-10 ADR comments:  {((counts >= 6) & (counts <= 10)).sum():,}")
    print(f"    11-20 ADR comments: {((counts >= 11) & (counts <= 20)).sum():,}")
    print(f"    20+ ADR comments:   {(counts > 20).sum():,}")
    print()

    adr_comments_df["subreddit_mapped"] = adr_comments_df["post_id"].map(posts_lookup)
    print("  --- By Subreddit ---")
    for sub in sorted(adr_per_post["subreddit"].unique()):
        n = (adr_comments_df["subreddit_mapped"] == sub).sum()
        sub_posts = adr_per_post[adr_per_post["subreddit"] == sub]
        avg = sub_posts["adr_comment_count"].mean() if len(sub_posts) else 0
        print(f"    r/{sub}: {n:,} ADR comments ({avg:.1f} avg/post)")
    print()

    print("  --- Top 10 Medications (in comments) ---")
    for med, n in adr_comments_df["medication"].value_counts().head(10).items():
        print(f"    {med}: {n:,}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ──────────────────────────────────────────────────────────────────────

def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Saved %s", path)


def save_adr_csv(adr_data: List[Dict], path: Path) -> None:
    """Save the enriched ADR mentions CSV with all required columns."""
    if not adr_data:
        return
    df = pd.DataFrame(adr_data)
    # Flatten list columns
    for col in ("adr_types", "symptoms"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)
    col_order = [
        "subreddit", "post_id", "comment_id", "medication", "adr_types",
        "symptoms", "severity", "sentiment", "context", "timestamp",
        "posted_time", "upvotes", "downvotes", "net_score", "num_comments",
        "source", "source_id", "author", "created_utc", "engagement_ratio",
    ]
    cols = [c for c in col_order if c in df.columns]
    df[cols].to_csv(path, index=False, encoding="utf-8")
    log.info("Saved %s  (%d rows)", path, len(df))


def save_engagement_csv(posts: List[Dict], path: Path) -> None:
    """Per-post engagement summary CSV."""
    if not posts:
        return
    df = pd.DataFrame(posts)
    cols = [
        "subreddit", "id", "title", "author", "created_utc", "posted_time",
        "score", "upvotes", "downvotes", "upvote_ratio", "num_comments",
        "engagement_ratio", "url",
    ]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(path, index=False, encoding="utf-8")
    log.info("Saved %s  (%d rows)", path, len(df))


def generate_summary_statistics(adr_data: List[Dict], path: Path) -> None:
    if not adr_data:
        return
    df = pd.DataFrame(adr_data)
    all_symptoms: List[str] = []
    for s in df["symptoms"]:
        if isinstance(s, list):
            all_symptoms.extend(s)
    all_types: List[str] = []
    for t in df["adr_types"]:
        if isinstance(t, list):
            all_types.extend(t)

    summary = {
        "total_adr_mentions": len(df),
        "unique_medications": int(df["medication"].nunique()),
        "date_range": {
            "start": str(df["created_utc"].min()),
            "end": str(df["created_utc"].max()),
        },
        "top_medications": df["medication"].value_counts().head(20).to_dict(),
        "sources": df["source"].value_counts().to_dict(),
        "severity_distribution": df["severity"].value_counts().to_dict(),
        "average_sentiment": float(df["sentiment"].mean()),
        "most_common_symptoms": dict(Counter(all_symptoms).most_common(30)),
        "most_common_adr_types": dict(Counter(all_types).most_common(10)),
    }
    save_json(summary, path)


# ──────────────────────────────────────────────────────────────────────
# VISUALIZATION UTILITIES
# ──────────────────────────────────────────────────────────────────────

def _setup_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
    })


def _clip_series(s: pd.Series, upper_pct: float = 97.5) -> pd.Series:
    """Clip to the upper_pct-th percentile to prevent off-scale plots."""
    if s.empty:
        return s
    cap = np.percentile(s.dropna(), upper_pct)
    return s.clip(upper=cap)


def _safe_savefig(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path.name)


# ──────────────────────────────────────────────────────────────────────
# PLOT GROUP A -- from reddit_adr_analyzer.py (improved)
# ──────────────────────────────────────────────────────────────────────

def plot_adr_timeline(df: pd.DataFrame, viz_dir: Path):
    """ADR mentions over time by top-10 medications."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 8))
    top_meds = df["medication"].value_counts().head(10).index
    df_top = df[df["medication"].isin(top_meds)].copy()
    df_top["ym"] = df_top["date"].dt.to_period("M")

    for med in top_meds:
        timeline = df_top[df_top["medication"] == med].groupby("ym").size()
        if timeline.empty:
            continue
        idx = timeline.index.to_timestamp()
        ax.plot(idx, timeline.values, marker="o", label=med, linewidth=2, markersize=4)

    ax.set_xlabel("Date")
    ax.set_ylabel("ADR Mentions")
    ax.set_title("ADR Mentions Over Time by Medication (Top 10)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _safe_savefig(fig, viz_dir / "adr_timeline.png")


def plot_adr_categories(df: pd.DataFrame, viz_dir: Path):
    """Horizontal bar chart of ADR category counts."""
    if df.empty:
        return
    all_types: List[str] = []
    for t in df["adr_types"]:
        if isinstance(t, list):
            all_types.extend(t)
    if not all_types:
        return
    counter = Counter(all_types).most_common(15)
    types, counts = zip(*counter)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(types)), counts, color=sns.color_palette("viridis", len(types)))
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    ax.set_xlabel("Mentions")
    ax.set_title("Most Common ADR Categories", fontweight="bold")
    ax.invert_yaxis()
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(c, i, f" {c}", va="center", fontsize=10)
    _safe_savefig(fig, viz_dir / "adr_categories.png")


def plot_symptom_heatmap(df: pd.DataFrame, viz_dir: Path):
    """Medication x Symptom heatmap."""
    if df.empty:
        return
    top_meds = df["medication"].value_counts().head(15).index.tolist()
    all_syms: List[str] = []
    for s in df["symptoms"]:
        if isinstance(s, list):
            all_syms.extend(s)
    top_syms = [s for s, _ in Counter(all_syms).most_common(20)]
    if not top_meds or not top_syms:
        return

    matrix = []
    for med in top_meds:
        med_df = df[df["medication"] == med]
        row = []
        for sym in top_syms:
            cnt = sum(1 for sl in med_df["symptoms"] if isinstance(sl, list) and sym in sl)
            row.append(cnt)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(matrix, xticklabels=top_syms, yticklabels=top_meds,
                cmap="YlOrRd", annot=True, fmt="d",
                cbar_kws={"label": "Count"}, ax=ax)
    ax.set_title("ADR Symptoms by Medication", fontweight="bold")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    _safe_savefig(fig, viz_dir / "symptom_heatmap.png")


def plot_severity_distribution(df: pd.DataFrame, viz_dir: Path):
    """Bar chart of severity levels."""
    if df.empty or df["severity"].dropna().empty:
        return
    sev = df["severity"].value_counts()
    colors_map = {"severe": "#d62728", "moderate": "#ff7f0e", "mild": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sev.index, sev.values,
                  color=[colors_map.get(s, "#7f7f7f") for s in sev.index],
                  edgecolor="black")
    ax.set_xlabel("Severity")
    ax.set_ylabel("ADR Mentions")
    ax.set_title("ADR Severity Distribution", fontweight="bold")
    for b, v in zip(bars, sev.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(v):,}",
                ha="center", va="bottom", fontsize=11)
    _safe_savefig(fig, viz_dir / "severity_distribution.png")


def plot_sentiment_boxplot(df: pd.DataFrame, viz_dir: Path):
    """Sentiment box-plot by top-10 medications."""
    if df.empty:
        return
    top10 = df["medication"].value_counts().head(10).index
    data_groups = [df.loc[df["medication"] == m, "sentiment"].dropna().values for m in top10]
    if not any(len(g) for g in data_groups):
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_groups, labels=top10, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Neutral")
    ax.set_xlabel("Medication")
    ax.set_ylabel("Sentiment (-1 to 1)")
    ax.set_title("Sentiment Distribution by Medication", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    _safe_savefig(fig, viz_dir / "sentiment_analysis.png")


def plot_monthly_trend(df: pd.DataFrame, viz_dir: Path):
    """Overall monthly ADR mention trend with fill."""
    if df.empty:
        return
    df2 = df.copy()
    df2["ym"] = df2["date"].dt.to_period("M")
    monthly = df2.groupby("ym").size()
    if monthly.empty:
        return
    idx = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(idx, monthly.values, marker="o", linewidth=2, markersize=5, color="#e74c3c")
    ax.fill_between(idx, monthly.values, alpha=0.3, color="#e74c3c")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total ADR Mentions")
    ax.set_title("Overall ADR Mentions Trend", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    _safe_savefig(fig, viz_dir / "monthly_trend.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT GROUP B -- from analyze_comments.py (improved)
# ──────────────────────────────────────────────────────────────────────

def plot_comment_adr_distribution(adr_df: pd.DataFrame, posts_df: pd.DataFrame, viz_dir: Path):
    """Histogram of ADR comments per post (capped at 97.5th pctl)."""
    adr_comments = adr_df[adr_df["source"] == "comment"]
    if adr_comments.empty:
        return
    per_post = adr_comments.groupby("post_id").size()
    capped = _clip_series(per_post)

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = min(30, int(capped.max()) + 1)
    ax.hist(capped, bins=bins, edgecolor="black", alpha=0.7, color="#e74c3c")
    ax.axvline(per_post.mean(), color="blue", ls="--", lw=2, label=f"Mean: {per_post.mean():.1f}")
    ax.axvline(per_post.median(), color="green", ls="--", lw=2, label=f"Median: {per_post.median():.1f}")
    ax.set_xlabel("ADR Comments per Post")
    ax.set_ylabel("Number of Posts")
    ax.set_title("Distribution of ADR Comments per Post", fontweight="bold")
    ax.legend()
    n_outliers = (per_post > capped.max()).sum()
    if n_outliers:
        ax.annotate(f"{n_outliers} posts with higher values clipped",
                    xy=(0.98, 0.95), xycoords="axes fraction", ha="right", fontsize=9, fontstyle="italic")
    _safe_savefig(fig, viz_dir / "adr_comments_distribution.png")


def plot_adr_by_subreddit_boxplot(adr_df: pd.DataFrame, posts_df: pd.DataFrame, viz_dir: Path):
    """Box-plot of ADR comment counts per post, grouped by subreddit."""
    adr_comments = adr_df[adr_df["source"] == "comment"]
    if adr_comments.empty:
        return
    per_post = adr_comments.groupby("post_id").size().reset_index(name="cnt")
    lookup = posts_df.set_index("id")["subreddit"].to_dict()
    per_post["subreddit"] = per_post["post_id"].map(lookup)
    per_post.dropna(subset=["subreddit"], inplace=True)
    if per_post.empty:
        return
    cap = np.percentile(per_post["cnt"], 97.5)
    per_post["cnt_c"] = per_post["cnt"].clip(upper=cap)
    order = per_post.groupby("subreddit")["cnt"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=per_post, x="subreddit", y="cnt_c", order=order, ax=ax,
                showfliers=False, palette="viridis")
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("ADR Comments per Post")
    ax.set_title("ADR Comments per Post by Subreddit (clipped at p97.5)", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    _safe_savefig(fig, viz_dir / "adr_comments_by_subreddit_boxplot.png")


def plot_avg_adr_by_subreddit(adr_df: pd.DataFrame, posts_df: pd.DataFrame, viz_dir: Path):
    """Bar chart: average ADR comments per subreddit."""
    adr_comments = adr_df[adr_df["source"] == "comment"]
    if adr_comments.empty:
        return
    per_post = adr_comments.groupby("post_id").size().reset_index(name="cnt")
    lookup = posts_df.set_index("id")["subreddit"].to_dict()
    per_post["subreddit"] = per_post["post_id"].map(lookup)
    per_post.dropna(subset=["subreddit"], inplace=True)
    stats = per_post.groupby("subreddit")["cnt"].mean().sort_values(ascending=False)
    if stats.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(stats.index, stats.values, color=sns.color_palette("viridis", len(stats)))
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("Avg ADR Comments per Post")
    ax.set_title("Average ADR Comments per Post by Subreddit", fontweight="bold")
    for b, v in zip(bars, stats.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    _safe_savefig(fig, viz_dir / "avg_adr_comments_by_subreddit.png")


def plot_adr_pie(adr_df: pd.DataFrame, viz_dir: Path):
    """Pie chart: ADR comment share by subreddit."""
    adr_comments = adr_df[adr_df["source"] == "comment"]
    if adr_comments.empty:
        return
    totals = adr_comments.groupby("subreddit").size().sort_values(ascending=False)
    if totals.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(totals.values, labels=totals.index, autopct="%1.1f%%",
           colors=sns.color_palette("viridis", len(totals)), pctdistance=0.8)
    ax.set_title("ADR Comments by Subreddit", fontweight="bold")
    _safe_savefig(fig, viz_dir / "adr_comments_pie.png")


def plot_top_medications_bar(adr_df: pd.DataFrame, viz_dir: Path):
    """Horizontal bar: top-15 medications in ADR comments."""
    adr_comments = adr_df[adr_df["source"] == "comment"]
    if adr_comments.empty:
        return
    top = adr_comments["medication"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top)), top.values, color=sns.color_palette("Reds_r", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("ADR Comments")
    ax.set_title("Top 15 Medications in ADR Comments", fontweight="bold")
    ax.invert_yaxis()
    for i, (b, c) in enumerate(zip(bars, top.values)):
        ax.text(c, i, f" {c:,}", va="center", fontsize=10)
    _safe_savefig(fig, viz_dir / "top_medications_adr_comments.png")


def plot_comment_severity(adr_df: pd.DataFrame, viz_dir: Path):
    """Bar chart of severity in ADR comments."""
    adr_comments = adr_df[adr_df["source"] == "comment"]
    if adr_comments.empty or adr_comments["severity"].dropna().empty:
        return
    sev = adr_comments["severity"].value_counts()
    cmap = {"severe": "#d62728", "moderate": "#ff7f0e", "mild": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sev.index, sev.values,
                  color=[cmap.get(s, "#7f7f7f") for s in sev.index], edgecolor="black")
    ax.set_xlabel("Severity")
    ax.set_ylabel("ADR Comments")
    ax.set_title("ADR Comment Severity Distribution", fontweight="bold")
    for b, v in zip(bars, sev.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(v):,}",
                ha="center", va="bottom", fontsize=11)
    _safe_savefig(fig, viz_dir / "adr_comments_severity.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT GROUP C -- NEW ENGAGEMENT & STATISTICAL PLOTS
# ──────────────────────────────────────────────────────────────────────

def plot_engagement_summary(posts_df: pd.DataFrame, viz_dir: Path):
    """Bar chart: min / max / mean / median of score and num_comments per post."""
    if posts_df.empty:
        return
    metrics = {}
    for col in ("score", "num_comments"):
        s = posts_df[col].dropna()
        if s.empty:
            continue
        metrics[col] = {
            "Min": s.min(), "Max": s.max(),
            "Mean": s.mean(), "Median": s.median(),
        }
    if not metrics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (col, vals) in zip(axes, metrics.items()):
        labels = list(vals.keys())
        values = list(vals.values())
        bars = ax.bar(labels, values, color=sns.color_palette("coolwarm", 4), edgecolor="black")
        ax.set_title(f"Post {col.replace('_', ' ').title()} Summary", fontweight="bold")
        ax.set_ylabel("Value")
        # Use symlog scale if range is large
        if max(values) > 100 * min(abs(v) for v in values if v != 0):
            ax.set_yscale("symlog", linthresh=1)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:,.1f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Engagement Metrics Summary", fontweight="bold", fontsize=14, y=1.02)
    _safe_savefig(fig, viz_dir / "engagement_summary.png")


def plot_avg_upvotes_per_year(posts_df: pd.DataFrame, viz_dir: Path):
    """Line chart: average post upvotes per year."""
    if posts_df.empty or "date" not in posts_df.columns:
        return
    df2 = posts_df.dropna(subset=["date"]).copy()
    df2["year"] = df2["date"].dt.year
    yearly = df2.groupby("year")["upvotes"].mean()
    if yearly.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(yearly.index, yearly.values, marker="s", linewidth=2, color="#2980b9", markersize=7)
    ax.fill_between(yearly.index, yearly.values, alpha=0.15, color="#2980b9")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Upvotes")
    ax.set_title("Average Post Upvotes per Year", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _safe_savefig(fig, viz_dir / "avg_upvotes_per_year.png")


def plot_comment_activity_over_time(comments_df: pd.DataFrame, viz_dir: Path):
    """Monthly histogram of comment creation."""
    if comments_df.empty:
        return
    df2 = comments_df.dropna(subset=["created_utc"]).copy()
    df2["date"] = pd.to_datetime(df2["created_utc"], unit="s", utc=True, errors="coerce")
    df2.dropna(subset=["date"], inplace=True)
    if df2.empty:
        return
    df2["ym"] = df2["date"].dt.to_period("M")
    monthly = df2.groupby("ym").size()
    idx = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(idx, monthly.values, width=25, color="#27ae60", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Comments")
    ax.set_title("Comment Activity Over Time", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.autofmt_xdate()
    _safe_savefig(fig, viz_dir / "comment_activity_over_time.png")


def plot_engagement_heatmap(posts_df: pd.DataFrame, viz_dir: Path):
    """Heatmap: year x subreddit of mean engagement_ratio."""
    if posts_df.empty or "date" not in posts_df.columns:
        return
    df2 = posts_df.dropna(subset=["date", "engagement_ratio"]).copy()
    if df2.empty:
        return
    df2["year"] = df2["date"].dt.year
    pivot = df2.pivot_table(index="subreddit", columns="year", values="engagement_ratio", aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)), max(6, len(pivot.index) * 0.6)))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": "Mean Engagement Ratio"}, ax=ax)
    ax.set_title("Engagement Heatmap (Year x Subreddit)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Subreddit")
    plt.yticks(rotation=0)
    _safe_savefig(fig, viz_dir / "engagement_heatmap.png")


def plot_upvotes_comments_boxplots(posts_df: pd.DataFrame, viz_dir: Path):
    """Side-by-side box-plots of upvotes and num_comments (outlier-safe)."""
    if posts_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, col, color, label in [
        (axes[0], "upvotes", "#3498db", "Upvotes"),
        (axes[1], "num_comments", "#e67e22", "Comments"),
    ]:
        s = posts_df[col].dropna()
        if s.empty:
            continue
        cap = np.percentile(s, 97.5)
        clipped = s.clip(upper=cap)
        bp = ax.boxplot(clipped, patch_artist=True, showfliers=True,
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_title(f"{label} Distribution (clipped p97.5)", fontweight="bold")
        ax.set_ylabel(label)
        stats_text = (
            f"Mean: {s.mean():.1f}\nMedian: {s.median():.1f}\n"
            f"p95: {np.percentile(s, 95):.0f}\nMax: {s.max():.0f}"
        )
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Post Engagement Distributions", fontweight="bold", fontsize=14)
    _safe_savefig(fig, viz_dir / "upvotes_comments_boxplots.png")


def plot_medication_engagement_trend(adr_df: pd.DataFrame, viz_dir: Path):
    """Dual-axis: ADR mention count + mean score over time for top medications."""
    if adr_df.empty or "date" not in adr_df.columns:
        return
    top5 = adr_df["medication"].value_counts().head(5).index
    df2 = adr_df[adr_df["medication"].isin(top5)].copy()
    df2["ym"] = df2["date"].dt.to_period("M")

    fig, axes = plt.subplots(len(top5), 1, figsize=(15, 4 * len(top5)), sharex=True)
    if len(top5) == 1:
        axes = [axes]

    for ax, med in zip(axes, top5):
        med_df = df2[df2["medication"] == med]
        grp = med_df.groupby("ym").agg(count=("medication", "size"), avg_score=("score", "mean"))
        if grp.empty:
            continue
        idx = grp.index.to_timestamp()

        color1 = "#2980b9"
        ax.bar(idx, grp["count"].values, width=20, color=color1, alpha=0.6, label="ADR mentions")
        ax.set_ylabel("ADR Mentions", color=color1)
        ax.tick_params(axis="y", labelcolor=color1)

        ax2 = ax.twinx()
        color2 = "#e74c3c"
        ax2.plot(idx, grp["avg_score"].values, color=color2, linewidth=2, marker="o",
                 markersize=4, label="Avg score")
        ax2.set_ylabel("Avg Score", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax.set_title(f"{med}", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Top Medication ADR Trends with Engagement", fontweight="bold", fontsize=14, y=1.01)
    fig.autofmt_xdate()
    _safe_savefig(fig, viz_dir / "medication_engagement_trend.png")


# ──────────────────────────────────────────────────────────────────────
# MASTER VISUALIZATION RUNNER
# ──────────────────────────────────────────────────────────────────────

def create_all_visualizations(
    adr_data: List[Dict],
    posts: List[Dict],
    comments: List[Dict],
    viz_dir: Path,
):
    """Run every visualisation and save to *viz_dir*."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    # Build DataFrames
    adr_df = pd.DataFrame(adr_data) if adr_data else pd.DataFrame()
    posts_df = pd.DataFrame(posts) if posts else pd.DataFrame()
    comments_df = pd.DataFrame(comments) if comments else pd.DataFrame()

    # Add date columns
    if not adr_df.empty and "created_utc" in adr_df.columns:
        adr_df["date"] = pd.to_datetime(adr_df["created_utc"], unit="s", utc=True, errors="coerce")
    if not posts_df.empty and "created_utc" in posts_df.columns:
        posts_df["date"] = pd.to_datetime(posts_df["created_utc"], unit="s", utc=True, errors="coerce")

    log.info("Creating visualisations in %s ...", viz_dir)

    # Group A -- original adr_analyzer plots
    plot_adr_timeline(adr_df, viz_dir)
    plot_adr_categories(adr_df, viz_dir)
    plot_symptom_heatmap(adr_df, viz_dir)
    plot_severity_distribution(adr_df, viz_dir)
    plot_sentiment_boxplot(adr_df, viz_dir)
    plot_monthly_trend(adr_df, viz_dir)

    # Group B -- original analyze_comments plots
    plot_comment_adr_distribution(adr_df, posts_df, viz_dir)
    plot_adr_by_subreddit_boxplot(adr_df, posts_df, viz_dir)
    plot_avg_adr_by_subreddit(adr_df, posts_df, viz_dir)
    plot_adr_pie(adr_df, viz_dir)
    plot_top_medications_bar(adr_df, viz_dir)
    plot_comment_severity(adr_df, viz_dir)

    # Group C -- new engagement plots
    plot_engagement_summary(posts_df, viz_dir)
    plot_avg_upvotes_per_year(posts_df, viz_dir)
    plot_comment_activity_over_time(comments_df, viz_dir)
    plot_engagement_heatmap(posts_df, viz_dir)
    plot_upvotes_comments_boxplots(posts_df, viz_dir)
    plot_medication_engagement_trend(adr_df, viz_dir)

    log.info("All visualisations saved.")


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reddit ADR Scraper & Analyser (unified pipeline v2)",
    )
    parser.add_argument(
        "--subreddits", "-s",
        default=",".join(DEFAULT_SUBREDDITS),
        help="Comma-separated list of subreddits (default: ADHD-related set)",
    )
    parser.add_argument(
        "--years", "-y",
        type=int, default=15,
        help="How many years of history to attempt collecting (default: 15)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="adr_analysis_output_v2",
        help="Output directory (default: adr_analysis_output_v2)",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip visualisation generation",
    )
    args = parser.parse_args()

    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"

    # ── Authenticate ──
    use_oauth, headers, base_endpoint = authenticate()

    # ── Collect ──
    print("=" * 60)
    print(f"  Reddit ADR Pipeline v2  |  Subreddits: {subreddits}")
    print(f"  Target history: {args.years} years  |  Output: {output_dir}")
    print("=" * 60)

    all_posts, all_comments = collect_all_data(
        subreddits, args.years, use_oauth, headers, base_endpoint,
    )

    # ── Save raw data ──
    save_json(all_posts,    output_dir / "raw_posts.json")
    save_json(all_comments, output_dir / "raw_comments.json")

    # ── ADR analysis ──
    adr_data = analyze_adr_mentions(all_posts, all_comments)
    save_json(adr_data, output_dir / "adr_mentions.json")
    save_adr_csv(adr_data, output_dir / "adr_mentions.csv")

    # ── Engagement CSV ──
    save_engagement_csv(all_posts, output_dir / "engagement_dataset.csv")

    # ── Summary statistics ──
    generate_summary_statistics(adr_data, output_dir / "summary_statistics.json")

    # ── Comment-level ADR statistics (console) ──
    posts_df = pd.DataFrame(all_posts) if all_posts else pd.DataFrame()
    adr_df = pd.DataFrame(adr_data) if adr_data else pd.DataFrame()
    compute_comment_adr_statistics(posts_df, adr_df)

    # ── Visualisations ──
    if not args.no_visualizations:
        create_all_visualizations(adr_data, all_posts, all_comments, viz_dir)

    # ── Done ──
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Posts collected:    {len(all_posts):,}")
    print(f"  Comments collected: {len(all_comments):,}")
    print(f"  ADR mentions:      {len(adr_data):,}")
    print(f"\n  Output directory: {output_dir}/")
    print(f"    raw_posts.json")
    print(f"    raw_comments.json")
    print(f"    adr_mentions.json")
    print(f"    adr_mentions.csv")
    print(f"    engagement_dataset.csv")
    print(f"    summary_statistics.json")
    if not args.no_visualizations:
        print(f"    visualizations/  ({len(list(viz_dir.glob('*.png')))} plots)")
    print("=" * 60)


if __name__ == "__main__":
    main()
