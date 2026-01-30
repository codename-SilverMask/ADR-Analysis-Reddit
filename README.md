# Reddit ADR Analyzer v2

A unified Python pipeline that scrapes Reddit posts and comments, detects Adverse Drug Reaction (ADR) mentions using NLP pattern matching, collects engagement metrics, performs sentiment and severity analysis, and generates comprehensive visualizations -- all in a single end-to-end run.

## Features

- **Automated Reddit Scraping** -- Collects posts and comments from multiple subreddits using OAuth
- **Historical Data Collection** -- Three-strategy approach (Pushshift + Reddit listings + search) targeting up to 15 years of data
- **Medication Detection** -- Identifies 186+ medications including brand names, generics, and slang terms
- **ADR Pattern Matching** -- Context-window NLP extraction around medication mentions across 8 ADR categories
- **Severity & Sentiment Scoring** -- Classifies ADR severity (severe/moderate/mild) and computes keyword-ratio sentiment
- **Engagement Metrics** -- Captures upvotes, downvotes, comment counts, and engagement ratios for every post
- **Comment-Level ADR Statistics** -- Computes per-post ADR comment distributions and subreddit breakdowns
- **18 Analytical Visualizations** -- Timelines, heatmaps, box plots, pie charts, engagement overlays, and more
- **Multi-format Export** -- JSON, CSV (ADR mentions + engagement), and summary statistics
- **DOCX Report Generation** -- Automated comprehensive report with all statistics, tables, and embedded graphs

## Installation

### Prerequisites

- Python 3.8+
- Reddit API credentials (OAuth2)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd redditPaginationClaude
```

2. Create and activate a virtual environment:

```bash
python -m venv ADRenv
# Windows:
ADRenv\Scripts\activate
# macOS/Linux:
source ADRenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure Reddit API credentials (choose one):
   - **Environment variables** (recommended):
     ```bash
     export REDDIT_CLIENT_ID="your_client_id"
     export REDDIT_CLIENT_SECRET="your_secret"
     export REDDIT_USERNAME="your_username"
     export REDDIT_PASSWORD="your_password"
     ```
   - **Edit `reddit_scraper.py`** directly (lines 54-57)

## Usage

### Run the full pipeline

```bash
python reddit_scraper.py
```

This scrapes all 13 default subreddits with 15 years of history.

### Custom subreddits

```bash
python reddit_scraper.py --subreddits ADHD,ADHDmemes,adhdwomen
```

The `--subreddits` flag **overrides** the default list entirely.

### Custom history window

```bash
python reddit_scraper.py --years 5
```

### Custom output directory

```bash
python reddit_scraper.py --output-dir my_analysis
```

### Skip visualizations

```bash
python reddit_scraper.py --no-visualizations
```

## Default Subreddits

```
ADHD, ADHDmemes, adhdwomen, adhd_anxiety,
ADHD_partners, ADHDUK, VyvanseADHD, AdhdRelationships,
drug, DrugAddiction, HowDrugsWork, pharms, drugaddicts
```

## Output Structure

```
adr_analysis_output_v2/
├── raw_posts.json              # All scraped posts with engagement data
├── raw_comments.json           # All scraped comments with scores
├── adr_mentions.json           # Extracted ADR records (full detail)
├── adr_mentions.csv            # ADR records with engagement columns
├── engagement_dataset.csv      # Per-post engagement summary
├── Posts.csv                   # Deduplicated posts with all metadata (26.4 MB)
├── Comments.csv                # Deduplicated comments with post IDs (195.4 MB)
├── summary_statistics.json     # Aggregated metrics
└── visualizations/             # 18 PNG charts
    ├── adr_timeline.png
    ├── adr_categories.png
    ├── symptom_heatmap.png
    ├── severity_distribution.png
    ├── sentiment_analysis.png
    ├── monthly_trend.png
    ├── adr_comments_distribution.png
    ├── adr_comments_by_subreddit_boxplot.png
    ├── avg_adr_comments_by_subreddit.png
    ├── adr_comments_pie.png
    ├── top_medications_adr_comments.png
    ├── adr_comments_severity.png
    ├── engagement_summary.png
    ├── avg_upvotes_per_year.png
    ├── comment_activity_over_time.png
    ├── engagement_heatmap.png
    ├── upvotes_comments_boxplots.png
    └── medication_engagement_trend.png
```

### CSV Descriptions

**`adr_mentions.csv`** -- One row per detected ADR mention. Columns:

| Column       | Description                                     |
| ------------ | ----------------------------------------------- |
| subreddit    | Source subreddit                                |
| post_id      | Reddit post ID                                  |
| comment_id   | Reddit comment ID (null for post-sourced)       |
| medication   | Detected medication name (lowercase)            |
| adr_types    | Semicolon-separated ADR categories              |
| symptoms     | Semicolon-separated symptom keywords            |
| severity     | severe, moderate, mild, or blank                |
| sentiment    | Keyword-ratio score (-1 to +1)                  |
| context      | ~400-char text window around medication mention |
| timestamp    | Human-readable UTC timestamp                    |
| posted_time  | Alternate timestamp format                      |
| upvotes      | Estimated upvotes                               |
| downvotes    | Estimated downvotes                             |
| net_score    | Net score (upvotes - downvotes)                 |
| num_comments | Comment count on parent post                    |

**`engagement_dataset.csv`** -- One row per scraped post. Columns:

| Column           | Description                                          |
| ---------------- | ---------------------------------------------------- |
| subreddit        | Source subreddit                                     |
| id               | Reddit post ID (foreign key to adr_mentions.post_id) |
| title            | Post title                                           |
| author           | Reddit username                                      |
| created_utc      | Unix epoch timestamp                                 |
| posted_time      | Human-readable UTC timestamp                         |
| score            | Net score from Reddit                                |
| upvotes          | Estimated upvote count                               |
| downvotes        | Estimated downvote count                             |
| upvote_ratio     | Reddit's reported upvote ratio (0-1)                 |
| num_comments     | Total comment count                                  |
| engagement_ratio | (score + num_comments) / post_age_hours              |
| url              | Direct URL to post                                   |

The two datasets are related through `engagement_dataset.id = adr_mentions.post_id` (one-to-many).

**`Posts.csv`** -- One row per unique post (21,630 rows, deduplicated by post ID). Columns:

| Column           | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| post_id          | Unique Reddit post ID (primary key, joins to Comments.csv)          |
| subreddit        | Source subreddit                                                    |
| title            | Post title (1 -- 312 chars)                                         |
| selftext         | Full post body text (empty for link/image posts; 14.1% empty)       |
| author           | Reddit username (15,943 unique authors)                             |
| created_utc      | Unix epoch timestamp                                                |
| posted_time      | Human-readable UTC timestamp                                        |
| score            | Net score (0 -- 30,610; mean 331.26, median 4)                      |
| upvotes          | Estimated upvote count (from score + upvote_ratio)                  |
| downvotes        | Estimated downvote count                                            |
| upvote_ratio     | Proportion of votes that are upvotes (0.10 -- 1.00)                 |
| num_comments     | Comment count only (not the comment text; 0 -- 2,600)               |
| engagement_ratio | Time-normalised engagement: (score + num_comments) / post_age_hours |
| url              | Direct Reddit URL to the post                                       |

**`Comments.csv`** -- One row per unique comment (510,266 rows, deduplicated by comment ID). Columns:

| Column      | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| comment_id  | Unique Reddit comment ID (primary key)                         |
| post_id     | Parent post ID (foreign key to Posts.csv; zero orphans)        |
| subreddit   | Source subreddit                                               |
| body        | Full comment text with Markdown (0 -- 9,998 chars; mean 310.9) |
| author      | Reddit username (135,260 unique authors)                       |
| created_utc | Unix epoch timestamp                                           |
| posted_time | Human-readable UTC timestamp                                   |
| score       | Net score (-265 -- 10,475; mean 15.22, median 3)               |
| upvotes     | Estimated upvotes: max(score, 0)                               |
| downvotes   | Estimated downvotes: max(-score, 0)                            |
| depth       | Nesting depth (0 = top-level; max 9; 46.7% are depth 0)        |

**Dataset relationships:**

```
Posts.csv (post_id)  ←  1 : N  →  Comments.csv (post_id)
Posts.csv (post_id)  ←  1 : N  →  adr_mentions.csv (post_id)
engagement_dataset.csv (id) = Posts.csv (post_id)  -- same posts, Posts.csv adds selftext
```

---

### Dataset Overview

| Metric                      | Value                             |
| --------------------------- | --------------------------------- |
| Total Posts Scraped         | 21,630                            |
| Total Comments Scraped      | 510,266                           |
| Subreddits Covered          | 13                                |
| Date Range                  | July 24, 2011 -- January 30, 2026 |
| Total ADR Mentions          | 56,704                            |
| ADR Mentions from Posts     | 19,805 (34.9%)                    |
| ADR Mentions from Comments  | 36,899 (65.1%)                    |
| Unique Medications Detected | 186                               |

### Posts Per Subreddit

| Subreddit           | Posts | % of Total |
| ------------------- | ----- | ---------- |
| r/ADHD              | 4,402 | 20.4%      |
| r/adhdwomen         | 3,745 | 17.3%      |
| r/VyvanseADHD       | 2,951 | 13.6%      |
| r/adhd_anxiety      | 2,715 | 12.6%      |
| r/ADHDUK            | 2,603 | 12.0%      |
| r/ADHD_partners     | 1,454 | 6.7%       |
| r/ADHDmemes         | 1,265 | 5.8%       |
| r/AdhdRelationships | 1,087 | 5.0%       |
| r/DrugAddiction     | 866   | 4.0%       |
| r/pharms            | 255   | 1.2%       |
| r/drug              | 140   | 0.6%       |
| r/drugaddicts       | 94    | 0.4%       |
| r/HowDrugsWork      | 53    | 0.2%       |

### Post Engagement Statistics

| Metric  | Score  | Comments | Upvotes |
| ------- | ------ | -------- | ------- |
| Minimum | 0      | 0        | 0       |
| Maximum | 30,610 | 2,600    | 31,248  |
| Mean    | 331.26 | 32.67    | 334.21  |
| Median  | 4.0    | 6.0      | --      |

**Percentile Breakdown:**

| Percentile | Score | Comments | Upvotes |
| ---------- | ----- | -------- | ------- |
| p25        | 1     | 2        | 1       |
| p50        | 4     | 6        | 4       |
| p75        | 26    | 19       | 27      |
| p90        | 335   | 64       | 342     |
| p95        | 1,139 | 127      | 1,163   |
| p99        | 5,791 | 432      | 5,908   |

The distribution is heavily right-skewed: the median score is 4 while the maximum reaches 30,610. This necessitated percentile clipping and log scales in the visualizations.

### Comment Statistics

| Metric                     | Value          |
| -------------------------- | -------------- |
| Total Comments             | 510,266        |
| Comment Score Range        | -265 to 10,475 |
| Mean Comment Score         | 15.22          |
| Comments per Post (Mean)   | 25.53          |
| Comments per Post (Median) | 7.0            |
| Comments per Post (Max)    | 200            |

### Top 20 Medications by ADR Mentions

| Rank | Medication      | Mentions | % of Total |
| ---- | --------------- | -------- | ---------- |
| 1    | Vyvanse         | 14,491   | 25.6%      |
| 2    | Adderall        | 7,456    | 13.1%      |
| 3    | Wellbutrin      | 3,690    | 6.5%       |
| 4    | Concerta        | 3,193    | 5.6%       |
| 5    | Ritalin         | 2,889    | 5.1%       |
| 6    | Strattera       | 1,938    | 3.4%       |
| 7    | Lexapro         | 1,612    | 2.8%       |
| 8    | Zoloft          | 1,555    | 2.7%       |
| 9    | Methylphenidate | 1,269    | 2.2%       |
| 10   | Prozac          | 1,224    | 2.2%       |
| 11   | Guanfacine      | 987      | 1.7%       |
| 12   | Xanax           | 809      | 1.4%       |
| 13   | Weed            | 734      | 1.3%       |
| 14   | Amphetamine     | 584      | 1.0%       |
| 15   | Sertraline      | 540      | 1.0%       |
| 16   | Atomoxetine     | 520      | 0.9%       |
| 17   | Lamictal        | 511      | 0.9%       |
| 18   | Bupropion       | 502      | 0.9%       |
| 19   | Effexor         | 473      | 0.8%       |
| 20   | Gabapentin      | 407      | 0.7%       |

### Severity Distribution

| Severity     | Count  | % of Classified |
| ------------ | ------ | --------------- |
| Moderate     | 10,777 | 56.2%           |
| Severe       | 5,933  | 30.9%           |
| Mild         | 2,488  | 13.0%           |
| Unclassified | 37,506 | (66.1% of all)  |

Of the 19,198 severity-classified mentions, moderate reports dominate. The majority of mentions (37,506) did not contain explicit severity-indicating language.

### ADR Category Distribution

| Category               | Mentions |
| ---------------------- | -------- |
| Psychological Symptoms | 30,182   |
| Side Effect Indicators | 17,698   |
| Physical Symptoms      | 16,206   |
| Withdrawal / Tolerance | 12,994   |
| Cardiovascular         | 2,707    |
| Cognitive              | 1,617    |
| Neurological           | 321      |
| Gastrointestinal       | 124      |

Psychological symptoms dominate because keywords like anxiety, depression, and mood swings are very common in ADHD community discussions.

### Top 20 Symptoms

| Symptom        | Mentions |
| -------------- | -------- |
| Anxiety        | 15,019   |
| Side effect    | 7,234    |
| Depression     | 6,070    |
| Made me        | 3,588    |
| Rash           | 3,347    |
| Crash          | 3,161    |
| Quit           | 2,860    |
| Anxious        | 2,434    |
| Tired          | 2,288    |
| Panic          | 1,915    |
| Withdrawal     | 1,829    |
| Emotional      | 1,601    |
| Itching        | 1,571    |
| Depressed      | 1,473    |
| Heart rate     | 1,387    |
| Blood pressure | 1,339    |
| Insomnia       | 1,316    |
| Concern        | 1,283    |
| Tolerance      | 1,252    |
| Rage           | 1,207    |

### ADR Comments Per Post

| Metric                      | Value |
| --------------------------- | ----- |
| Posts with >= 1 ADR comment | 7,155 |
| Minimum                     | 1     |
| Maximum                     | 131   |
| Mean                        | 5.16  |
| Median                      | 3.0   |
| Std Dev                     | 7.81  |

**Distribution:**

| Range     | Count | %     |
| --------- | ----- | ----- |
| Exactly 1 | 2,279 | 31.8% |
| 2 -- 5    | 3,011 | 42.1% |
| 6 -- 10   | 1,002 | 14.0% |
| 11 -- 20  | 589   | 8.2%  |
| 20+       | 274   | 3.8%  |

### Engagement Metrics

| Metric                  | Value  |
| ----------------------- | ------ |
| Engagement Ratio Min    | 0.0000 |
| Engagement Ratio Max    | 159.43 |
| Engagement Ratio Mean   | 0.2488 |
| Engagement Ratio Median | 0.0060 |
| Mean Upvote Ratio       | 92.8%  |

### Average Sentiment Score: -1.0

The uniformly negative sentiment confirms that all detected mentions occur in adverse reaction contexts, as the pipeline filters out purely positive mentions.

---

## Visualizations

<img width="3930" height="5037" alt="Image" src="https://github.com/user-attachments/assets/4d4b1215-cb2e-4b7d-9a67-bebc3f7eb019" />
<img width="3505" height="1525" alt="Image" src="https://github.com/user-attachments/assets/dc6b2cf4-54ec-4818-baef-3a50bdbb2e48" />
<img width="3030" height="1867" alt="Image" src="https://github.com/user-attachments/assets/9a23f82f-98ea-421d-a38d-99368d752f6c" />
<img width="2601" height="1643" alt="Image" src="https://github.com/user-attachments/assets/4e3d67ea-2271-4fd4-bb86-954a1eb6da66" />
<img width="3498" height="2767" alt="Image" src="https://github.com/user-attachments/assets/774638af-2138-4b2e-adec-096acffd8503" />
<img width="3221" height="2104" alt="Image" src="https://github.com/user-attachments/assets/10d16385-9e40-4234-ae63-5b11ea32bcd5" />
<img width="3970" height="1959" alt="Image" src="https://github.com/user-attachments/assets/ca6e8f7a-e366-411e-ae2f-63323d40a928" />
<img width="3389" height="2104" alt="Image" src="https://github.com/user-attachments/assets/2ec02624-d2fe-40a4-81ea-2419493241b9" />
<img width="3493" height="2362" alt="Image" src="https://github.com/user-attachments/assets/026e26f8-b6e2-4004-9d0d-0a420b858ad2" />
<img width="3040" height="1642" alt="Image" src="https://github.com/user-attachments/assets/d22ae045-482e-4c8d-b06b-4b8f91b437a8" />
<img width="2613" height="2434" alt="Image" src="https://github.com/user-attachments/assets/6eb756a8-6fe7-4c16-b2d6-54ff9a64b1b4" />
<img width="2575" height="1643" alt="Image" src="https://github.com/user-attachments/assets/b2f9b20b-6ed9-4205-8548-1adfe9f2c0e8" />
<img width="4585" height="1934" alt="Image" src="https://github.com/user-attachments/assets/a1207506-4b19-40ad-a6f7-e48d790867f9" />
<img width="2965" height="1900" alt="Image" src="https://github.com/user-attachments/assets/8205974c-fcb6-4f9c-bb90-1d30203d1efa" />
<img width="3506" height="1641" alt="Image" src="https://github.com/user-attachments/assets/70e1b473-b9e6-4987-9b32-fa9a098b7324" />
<img width="3531" height="1525" alt="Image" src="https://github.com/user-attachments/assets/ac82f3f2-eb25-4d7c-a8d3-a8b45f1fcd35" />
<img width="3898" height="2058" alt="Image" src="https://github.com/user-attachments/assets/45e5ddfc-2efb-4627-8dcc-9c31d86c64cd" />
<img width="3939" height="1770" alt="Image" src="https://github.com/user-attachments/assets/4f81b325-b0f7-444f-a40e-0caf5925cf60" />

## Technical Details

### Data Collection Strategy

The pipeline uses three strategies to maximise temporal coverage:

1. **Pushshift API** -- Walks the full time window in chronological pages. Auto-detected at startup; gracefully skipped if unavailable.
2. **Reddit Listing Endpoints** -- Queries `/new`, `/top`, `/hot`, `/rising` with `t=all` per subreddit (~1,000 posts per sort).
3. **Reddit Search API** -- Medication-keyword queries (17 terms) to surface older relevant posts.

All results are deduplicated by post ID. Comments are fetched recursively per post via the Reddit API.

### NLP Pattern Matching

- **Medication regex**: Pre-compiled word-boundary patterns for 186+ terms
- **Context window**: +/-200 characters around each medication mention
- **ADR detection**: 8-category keyword scan (psychological, physical, side effect indicators, withdrawal/tolerance, cardiovascular, cognitive, gastrointestinal, neurological)
- **Severity classification**: Ordered check (severe -> moderate -> mild) with first-match assignment
- **Positive filter**: 50+ positive-indicator keywords to suppress false positives
- **Sentiment**: `(positive_count - negative_count) / total_count`

### Engagement Metrics

- **Upvotes/downvotes**: Estimated from Reddit's `score` and `upvote_ratio`
- **Engagement ratio**: `(score + num_comments) / post_age_hours`

### Rate Limiting

- 2-second delay between listing requests
- 1-second delay between comment fetches
- 2-second delay between search requests
- 1-second delay between Pushshift requests
- Automatic retry with back-off on HTTP 429

### Plot Improvements (vs v1)

- 97.5th percentile clipping on histograms and box plots
- Symlog scale where value ranges span orders of magnitude
- Outlier count annotations on clipped charts
- `showfliers=False` on box plots to prevent off-scale rendering

## Supported Medication Categories

- **Stimulants** (16): Adderall, Vyvanse, Ritalin, Concerta, etc.
- **Non-stimulants** (11): Strattera, Wellbutrin, Intuniv, etc.
- **Antidepressants** (40+): Prozac, Zoloft, Lexapro, Cymbalta, etc.
- **Anxiety Medications** (24+): Xanax, Ativan, Buspar, etc.
- **Mood Stabilizers/Antipsychotics** (15): Abilify, Seroquel, Lithium, etc.
- **Supplements/Other**: Modafinil, Armodafinil, etc.
- **Slang Terms** (30+): Addy, happy pills, brain meds, study meds, etc.
- **Alternative Substances** (35+): Cannabis, psilocybin, LSD terms

## Known Limitations

1. **Keyword-based NLP** -- May produce false positives (e.g., "gas" matching as a medication) and cannot capture nuanced or novel ADR descriptions
2. **Sentiment simplicity** -- Keyword-ratio scoring does not handle sarcasm, negation, or context-dependent language
3. **Self-selection bias** -- Reddit users experiencing side effects are more motivated to post
4. **Historical coverage** -- Depends on Pushshift availability; older data may be underrepresented
5. **Vote estimation** -- Reddit fuzzes upvote/downvote counts; estimates are approximations
6. **Memory-bound** -- All data held in memory; very large datasets may require streaming

## Disclaimer

This tool is for research and educational purposes only. The data collected should not be used for medical diagnosis or treatment decisions. Always consult healthcare professionals for medical advice.

## Support

For issues, questions, or suggestions, please open an issue in the repository.
