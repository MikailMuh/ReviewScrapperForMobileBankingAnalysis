"""
=============================================================================
Mobile Banking App Review Mining Pipeline
Steps 1-4: Scraping → Preprocessing → Classification → Sentiment Analysis
=============================================================================
Target Apps:
  - BCA Mobile        (com.bca)               → Traditional Bank
  - Jago              (com.jago.digitalBanking) → Digital Bank (Independent)
  - Blu by BCA Digital (com.bcadigital.blu)    → Digital Bank (BCA Subsidiary)

Requirements (install first):
  pip install google-play-scraper pandas numpy Sastrawi transformers torch tqdm

Note on App Store:
  app-store-scraper for iOS is often blocked/rate-limited.
  This pipeline focuses on Google Play Store which has the most Indonesian reviews.
  If you need App Store data, use the `app_store_scraper` library separately.
=============================================================================
"""

import os
import re
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
APPS = {
    "BCA Mobile": {
        "app_id": "com.bca",
        "category": "traditional_bank"
    },
    "Jago": {
        "app_id": "com.jago.digitalBanking",
        "category": "digital_bank_independent"
    },
    "Blu by BCA Digital": {
        "app_id": "com.bcadigital.blu",
        "category": "digital_bank_subsidiary"
    }
}

OUTPUT_DIR = "output"
MIN_REVIEW_WORDS = 10
REVIEWS_PER_APP = 5000  # target minimum per app
LANG_FILTER = "id"      # Bahasa Indonesia

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# STEP 1: DATA COLLECTION (SCRAPING)
# =============================================================
def step1_scrape_reviews():
    """
    Scrape reviews from Google Play Store for all 3 apps.
    Saves raw data to output/step1_raw_reviews.csv
    """
    from google_play_scraper import Sort, reviews, app as app_info

    print("=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    all_reviews = []

    for app_name, config in APPS.items():
        app_id = config["app_id"]
        category = config["category"]

        print(f"\n[*] Scraping: {app_name} ({app_id})")
        print(f"    Category: {category}")
        print(f"    Target: {REVIEWS_PER_APP} reviews")

        # Get app info
        try:
            info = app_info(app_id, lang="id", country="id")
            print(f"    Current rating: {info.get('score', 'N/A')}")
            print(f"    Total reviews: {info.get('ratings', 'N/A')}")
        except Exception as e:
            print(f"    [!] Could not fetch app info: {e}")

        # Scrape reviews in batches
        collected = []
        continuation_token = None
        batch_size = 200  # max per request
        max_retries = 3

        while len(collected) < REVIEWS_PER_APP:
            for attempt in range(max_retries):
                try:
                    result, continuation_token = reviews(
                        app_id,
                        lang="id",
                        country="id",
                        sort=Sort.NEWEST,
                        count=batch_size,
                        continuation_token=continuation_token
                    )
                    break
                except Exception as e:
                    print(f"    [!] Attempt {attempt + 1} failed: {e}")
                    time.sleep(5 * (attempt + 1))  # backoff
                    if attempt == max_retries - 1:
                        print(f"    [!] Giving up after {max_retries} attempts")
                        result = []

            if not result:
                print(f"    [!] No more reviews available. Got {len(collected)} total.")
                break

            for r in result:
                collected.append({
                    "app_name": app_name,
                    "app_id": app_id,
                    "category": category,
                    "review_id": r.get("reviewId", ""),
                    "username": r.get("userName", ""),
                    "review_text": r.get("content", ""),
                    "star_rating": r.get("score", 0),
                    "thumbs_up": r.get("thumbsUpCount", 0),
                    "review_date": r.get("at", ""),
                    "app_version": r.get("appVersion", ""),
                    "reply_text": r.get("replyContent", ""),
                    "reply_date": r.get("repliedAt", ""),
                    "platform": "google_play"
                })

            print(f"    Collected: {len(collected)} / {REVIEWS_PER_APP}", end="\r")
            time.sleep(1)  # rate limiting

        all_reviews.extend(collected)
        print(f"\n    [✓] Done: {len(collected)} reviews for {app_name}")

    # Save raw data
    df_raw = pd.DataFrame(all_reviews)
    raw_path = os.path.join(OUTPUT_DIR, "step1_raw_reviews.csv")
    df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"STEP 1 COMPLETE")
    print(f"Total raw reviews: {len(df_raw)}")
    print(f"Saved to: {raw_path}")
    print(f"{'=' * 60}\n")

    return df_raw


# =============================================================
# STEP 2: DATA PREPROCESSING
# =============================================================

# Indonesian slang dictionary (expandable)
SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "g": "tidak",
    "gaada": "tidak ada", "gada": "tidak ada",
    "bgt": "banget", "bngt": "banget",
    "tdk": "tidak", "gx": "tidak", "kagak": "tidak",
    "emg": "memang", "emang": "memang",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "blom": "belum",
    "yg": "yang", "dg": "dengan", "dgn": "dengan",
    "utk": "untuk", "bwt": "buat", "bt": "buat",
    "krn": "karena", "karna": "karena", "krna": "karena",
    "tp": "tapi", "tpi": "tapi",
    "sm": "sama", "sma": "sama",
    "lg": "lagi", "lgi": "lagi",
    "bs": "bisa", "bsa": "bisa",
    "msh": "masih", "masi": "masih",
    "mksd": "maksud", "mksud": "maksud",
    "knp": "kenapa", "knpa": "kenapa",
    "gmn": "gimana", "gmna": "gimana",
    "trs": "terus", "trus": "terus",
    "hrs": "harus",
    "jgn": "jangan", "jng": "jangan",
    "bkn": "bukan",
    "aja": "saja", "aj": "saja",
    "dr": "dari",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau",
    "sy": "saya", "aq": "saya", "ak": "saya", "gw": "saya", "gue": "saya",
    "lu": "kamu", "lo": "kamu",
    "org": "orang", "ornag": "orang",
    "bner": "benar", "bnr": "benar",
    "bnyk": "banyak", "byk": "banyak",
    "sgt": "sangat", "sngat": "sangat",
    "tlg": "tolong", "tlng": "tolong",
    "pdhl": "padahal", "pdahal": "padahal",
    "smpe": "sampai", "sampe": "sampai",
    "dpt": "dapat",
    "ngga": "tidak", "nggak": "tidak", "engga": "tidak", "enggak": "tidak",
    "lemot": "lambat",
    "ngebug": "error", "nge-bug": "error",
    "force close": "force close",
    "mantap": "bagus", "mantep": "bagus",
    "jos": "bagus",
    "pke": "pakai", "pk": "pakai",
    "dlu": "dulu", "dl": "dulu",
    "skrg": "sekarang", "skrang": "sekarang",
    "bgus": "bagus", "bgs": "bagus",
    "jlk": "jelek", "jlek": "jelek",
}

# Indonesian stopwords (common ones)
STOPWORDS_ID = set([
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "adalah",
    "dari", "ke", "ya", "akan", "juga", "atau", "ada", "saya", "sudah",
    "bisa", "tidak", "apa", "bukan", "jadi", "kalau", "karena", "saat",
    "oleh", "kami", "kita", "mereka", "dia", "ia", "anda", "kamu",
    "sangat", "lebih", "hanya", "tapi", "jika", "maka", "lagi", "pun",
    "bahwa", "sekali", "lain", "tersebut", "hal", "saja", "seperti",
    "masih", "belum", "telah", "dapat", "harus", "bagi", "agar",
    "punya", "nya", "dong", "sih", "kok", "deh", "loh", "lah", "kan",
    "se", "ter", "ber", "me", "men", "per", "pen", "di",
    "the", "is", "and", "to", "of", "a", "an", "in", "it", "i", "my",
    "this", "that", "was", "for", "but", "app", "very",
])


def clean_text(text):
    """Clean and normalize a single review text."""
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove emojis (keep Indonesian chars)
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_slang(text):
    """Replace Indonesian slang with standard words."""
    words = text.split()
    normalized = []
    for word in words:
        normalized.append(SLANG_DICT.get(word, word))
    return " ".join(normalized)


def remove_stopwords(text):
    """Remove Indonesian and English stopwords."""
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS_ID and len(w) > 1]
    return " ".join(filtered)


def count_words(text):
    """Count words in text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def step2_preprocess(df):
    """
    Preprocess raw reviews:
    1. Remove duplicates
    2. Filter language (basic: keep reviews with Indonesian characters)
    3. Clean text
    4. Normalize slang
    5. Remove stopwords (for tokenized version)
    6. Filter by minimum word count
    """
    print("=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    initial_count = len(df)
    print(f"Initial reviews: {initial_count}")

    # 2.1 Remove duplicates
    df = df.drop_duplicates(subset=["review_text"], keep="first")
    print(f"After dedup: {len(df)} (removed {initial_count - len(df)})")

    # 2.2 Remove empty reviews
    df = df[df["review_text"].notna() & (df["review_text"].str.strip() != "")]
    print(f"After removing empty: {len(df)}")

    # 2.3 Clean text
    df["clean_text"] = df["review_text"].apply(clean_text)
    print("[✓] Text cleaned (lowercased, URLs/emojis removed)")

    # 2.4 Normalize slang
    df["clean_text"] = df["clean_text"].apply(normalize_slang)
    print("[✓] Slang normalized")

    # 2.5 Create tokenized version (with stopword removal)
    df["tokenized_text"] = df["clean_text"].apply(remove_stopwords)
    print("[✓] Stopwords removed (tokenized version)")

    # 2.6 Filter by minimum word count (on clean_text, before stopword removal)
    df["word_count"] = df["clean_text"].apply(count_words)
    before_filter = len(df)
    df = df[df["word_count"] >= MIN_REVIEW_WORDS]
    print(f"After min {MIN_REVIEW_WORDS}-word filter: {len(df)} (removed {before_filter - len(df)})")

    # 2.7 Filter by date (last 12 months)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    cutoff_date = datetime.now() - timedelta(days=365)
    before_date = len(df)
    df = df[df["review_date"] >= cutoff_date]
    print(f"After 12-month filter: {len(df)} (removed {before_date - len(df)})")

    # Reset index
    df = df.reset_index(drop=True)

    # Save
    preprocess_path = os.path.join(OUTPUT_DIR, "step2_preprocessed.csv")
    df.to_csv(preprocess_path, index=False, encoding="utf-8-sig")

    # Summary per app
    print(f"\nPer-app summary:")
    for name in APPS:
        count = len(df[df["app_name"] == name])
        print(f"  {name}: {count} reviews")

    print(f"\n{'=' * 60}")
    print(f"STEP 2 COMPLETE")
    print(f"Total preprocessed reviews: {len(df)}")
    print(f"Saved to: {preprocess_path}")
    print(f"{'=' * 60}\n")

    return df


# =============================================================
# STEP 3: THEME CLASSIFICATION
# =============================================================

# Keyword lists for theme classification
PERFORMANCE_KEYWORDS = [
    "lambat", "lemot", "lag", "lama", "loading", "load",
    "crash", "error", "bug", "hang", "freeze", "force close",
    "cepat", "kencang", "smooth", "lancar", "responsif", "responsive",
    "patah", "putus", "stuck", "macet", "ngelag", "ngehang",
    "mati", "tutup sendiri", "keluar sendiri", "restart",
    "berat", "ringan", "stabil", "tidak stabil",
    "update", "versi", "terbaru", "downgrade",
    "login", "masuk", "otp", "fingerprint", "biometrik", "face id",
    "koneksi", "timeout", "gagal", "failed",
    "server", "maintenance", "gangguan", "down",
]

TRUST_KEYWORDS = [
    "aman", "keamanan", "security", "secure",
    "percaya", "trust", "terpercaya",
    "hack", "bobol", "retas", "curi",
    "takut", "khawatir", "was-was", "ragu",
    "hilang", "raib", "potong", "kepotong",
    "penipuan", "tipu", "scam", "fraud",
    "recommend", "rekomendasikan", "rekomendasi",
    "uninstall", "hapus", "delete", "cabut",
    "pindah", "ganti bank", "beralih",
    "bagus", "terbaik", "best", "top",
    "nyaman", "tenang", "yakin",
    "privasi", "data", "informasi pribadi",
    "saldo", "berkurang", "terdebet",
]

EFFICIENCY_KEYWORDS = [
    "ribet", "rumit", "complicated", "susah",
    "gampang", "mudah", "simple", "simpel", "praktis",
    "cepat", "instan", "quick", "fast",
    "lama", "makan waktu", "butuh waktu",
    "bingung", "confusing", "membingungkan",
    "jelas", "clear", "intuitive", "intuitif",
    "navigasi", "menu", "tombol", "button",
    "transfer", "bayar", "kirim", "tarik",
    "langkah", "step", "proses", "prosedur",
    "fitur", "feature", "fungsi",
    "ui", "ux", "tampilan", "interface", "desain",
    "user friendly", "ramah pengguna",
    "notifikasi", "notification",
    "saldo", "cek saldo", "mutasi", "riwayat",
]


def classify_themes(text):
    """Classify a review into performance, trust, and/or efficiency themes."""
    if not isinstance(text, str):
        return {"performance": False, "trust": False, "efficiency": False}

    text_lower = text.lower()

    themes = {
        "performance": any(kw in text_lower for kw in PERFORMANCE_KEYWORDS),
        "trust": any(kw in text_lower for kw in TRUST_KEYWORDS),
        "efficiency": any(kw in text_lower for kw in EFFICIENCY_KEYWORDS),
    }

    return themes


def step3_classify(df):
    """
    Classify each review into themes: performance, trust, efficiency.
    A review can belong to multiple themes.
    """
    print("=" * 60)
    print("STEP 3: THEME CLASSIFICATION")
    print("=" * 60)

    # Classify using clean_text (before stopword removal, to catch keywords)
    themes_list = df["clean_text"].apply(classify_themes)

    df["theme_performance"] = themes_list.apply(lambda x: x["performance"])
    df["theme_trust"] = themes_list.apply(lambda x: x["trust"])
    df["theme_efficiency"] = themes_list.apply(lambda x: x["efficiency"])

    # Count
    perf_count = df["theme_performance"].sum()
    trust_count = df["theme_trust"].sum()
    eff_count = df["theme_efficiency"].sum()
    no_theme = len(df[~(df["theme_performance"] | df["theme_trust"] | df["theme_efficiency"])])

    print(f"\nClassification results:")
    print(f"  Performance-related: {perf_count} ({perf_count/len(df)*100:.1f}%)")
    print(f"  Trust-related:       {trust_count} ({trust_count/len(df)*100:.1f}%)")
    print(f"  Efficiency-related:  {eff_count} ({eff_count/len(df)*100:.1f}%)")
    print(f"  No theme matched:    {no_theme} ({no_theme/len(df)*100:.1f}%)")

    # Per-app breakdown
    print(f"\nPer-app theme breakdown:")
    for name in APPS:
        app_df = df[df["app_name"] == name]
        n = len(app_df)
        if n == 0:
            continue
        p = app_df["theme_performance"].sum()
        t = app_df["theme_trust"].sum()
        e = app_df["theme_efficiency"].sum()
        print(f"  {name} (n={n}):")
        print(f"    Performance: {p} ({p/n*100:.1f}%)")
        print(f"    Trust:       {t} ({t/n*100:.1f}%)")
        print(f"    Efficiency:  {e} ({e/n*100:.1f}%)")

    # Save
    classify_path = os.path.join(OUTPUT_DIR, "step3_classified.csv")
    df.to_csv(classify_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"STEP 3 COMPLETE")
    print(f"Saved to: {classify_path}")
    print(f"{'=' * 60}\n")

    return df


# =============================================================
# STEP 4: SENTIMENT ANALYSIS
# =============================================================
def step4_sentiment(df):
    """
    Perform sentiment analysis using IndoBERT or fallback to
    simple rule-based approach if transformers/GPU not available.

    Tries IndoBERT first. If it fails (no GPU, OOM, etc.),
    falls back to a keyword-based sentiment scorer.
    """
    print("=" * 60)
    print("STEP 4: SENTIMENT ANALYSIS")
    print("=" * 60)

    use_indobert = False

    # Try loading IndoBERT
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "mdhugol/indonesia-bert-sentiment-classification"
        print(f"[*] Loading IndoBERT model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"[*] Device: {device_name}")

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            truncation=True
        )
        use_indobert = True
        print("[✓] IndoBERT loaded successfully!")

    except Exception as e:
        print(f"[!] IndoBERT failed to load: {e}")
        print("[*] Falling back to rule-based sentiment analysis...")

    if use_indobert:
        # ── IndoBERT-based sentiment ──
        print(f"\n[*] Running IndoBERT sentiment on {len(df)} reviews...")

        sentiments = []
        scores = []
        batch_size = 32

        for i in tqdm(range(0, len(df), batch_size), desc="Analyzing"):
            batch_texts = df["clean_text"].iloc[i:i+batch_size].tolist()

            # Truncate long texts
            batch_texts = [t[:512] if len(t) > 512 else t for t in batch_texts]

            try:
                results = sentiment_pipeline(batch_texts)
                for r in results:
                    label = r["label"].lower()
                    # Normalize labels
                    if "positif" in label or "positive" in label:
                        sentiments.append("positive")
                    elif "negatif" in label or "negative" in label:
                        sentiments.append("negative")
                    else:
                        sentiments.append("neutral")
                    scores.append(r["score"])
            except Exception as e:
                # If batch fails, assign neutral
                for _ in batch_texts:
                    sentiments.append("neutral")
                    scores.append(0.5)

        df["sentiment"] = sentiments
        df["sentiment_score"] = scores

    else:
        # ── Rule-based fallback ──
        print("[*] Using rule-based sentiment analysis...")

        POSITIVE_WORDS = set([
            "bagus", "baik", "mantap", "keren", "hebat", "suka", "senang",
            "puas", "nyaman", "aman", "cepat", "lancar", "smooth", "mudah",
            "gampang", "simple", "praktis", "membantu", "recommended",
            "terbaik", "top", "oke", "okay", "nice", "good", "great",
            "love", "perfect", "excellent", "amazing", "stabil", "responsif",
            "terima kasih", "makasih", "thanks", "bintang", "sempurna",
        ])

        NEGATIVE_WORDS = set([
            "jelek", "buruk", "lambat", "lemot", "lag", "crash", "error",
            "bug", "hang", "gagal", "susah", "ribet", "rumit", "bingung",
            "kecewa", "parah", "sampah", "busuk", "payah", "ancur",
            "tidak bisa", "force close", "stuck", "macet", "lama",
            "uninstall", "hapus", "pindah", "bohong", "tipu", "scam",
            "hilang", "bobol", "hack", "takut", "khawatir", "rugi",
            "mending", "kapok", "males", "bad", "worst", "terrible",
            "slow", "useless", "annoying", "frustrating",
        ])

        def rule_based_sentiment(text):
            if not isinstance(text, str):
                return "neutral", 0.5

            words = set(text.lower().split())
            pos_score = len(words & POSITIVE_WORDS)
            neg_score = len(words & NEGATIVE_WORDS)

            # Also factor in star rating
            total = pos_score + neg_score
            if total == 0:
                return "neutral", 0.5
            elif pos_score > neg_score:
                confidence = min(0.5 + (pos_score - neg_score) / (total * 2), 0.95)
                return "positive", confidence
            elif neg_score > pos_score:
                confidence = min(0.5 + (neg_score - pos_score) / (total * 2), 0.95)
                return "negative", confidence
            else:
                return "neutral", 0.5

        results = df["clean_text"].apply(rule_based_sentiment)
        df["sentiment"] = results.apply(lambda x: x[0])
        df["sentiment_score"] = results.apply(lambda x: x[1])

        # Boost with star rating
        # 1-2 stars: bias negative, 4-5 stars: bias positive
        def boost_with_rating(row):
            if row["star_rating"] <= 2 and row["sentiment"] == "neutral":
                return "negative"
            elif row["star_rating"] >= 4 and row["sentiment"] == "neutral":
                return "positive"
            return row["sentiment"]

        df["sentiment"] = df.apply(boost_with_rating, axis=1)

    # ── Results summary ──
    print(f"\nOverall sentiment distribution:")
    dist = df["sentiment"].value_counts()
    for s, c in dist.items():
        print(f"  {s}: {c} ({c/len(df)*100:.1f}%)")

    print(f"\nPer-app sentiment breakdown:")
    for name in APPS:
        app_df = df[df["app_name"] == name]
        n = len(app_df)
        if n == 0:
            continue
        print(f"\n  {name} (n={n}):")
        print(f"    Avg star rating: {app_df['star_rating'].mean():.2f}")
        for s in ["positive", "neutral", "negative"]:
            c = len(app_df[app_df["sentiment"] == s])
            print(f"    {s}: {c} ({c/n*100:.1f}%)")

    # ── Theme-specific sentiment ──
    print(f"\nTheme-specific sentiment (all apps):")
    for theme in ["performance", "trust", "efficiency"]:
        col = f"theme_{theme}"
        theme_df = df[df[col] == True]
        n = len(theme_df)
        if n == 0:
            continue
        print(f"\n  [{theme.upper()}] (n={n}):")
        for s in ["positive", "neutral", "negative"]:
            c = len(theme_df[theme_df["sentiment"] == s])
            print(f"    {s}: {c} ({c/n*100:.1f}%)")

    # Save
    final_path = os.path.join(OUTPUT_DIR, "step4_sentiment_final.csv")
    df.to_csv(final_path, index=False, encoding="utf-8-sig")

    # Also save a summary JSON
    summary = {
        "pipeline_date": datetime.now().isoformat(),
        "total_reviews": len(df),
        "apps": {}
    }
    for name in APPS:
        app_df = df[df["app_name"] == name]
        n = len(app_df)
        if n == 0:
            continue
        summary["apps"][name] = {
            "total_reviews": n,
            "category": APPS[name]["category"],
            "avg_star_rating": round(app_df["star_rating"].mean(), 2),
            "sentiment": {
                "positive": int(len(app_df[app_df["sentiment"] == "positive"])),
                "neutral": int(len(app_df[app_df["sentiment"] == "neutral"])),
                "negative": int(len(app_df[app_df["sentiment"] == "negative"])),
            },
            "themes": {
                "performance": int(app_df["theme_performance"].sum()),
                "trust": int(app_df["theme_trust"].sum()),
                "efficiency": int(app_df["theme_efficiency"].sum()),
            }
        }

    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"STEP 4 COMPLETE")
    print(f"Final dataset: {final_path}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}\n")

    return df


# =============================================================
# MAIN PIPELINE
# =============================================================
def run_pipeline(skip_scraping=False):
    """
    Run the full pipeline Steps 1-4.

    Args:
        skip_scraping: If True, load existing raw data from CSV
                       (useful if you already scraped).
    """
    print("\n" + "█" * 60)
    print("  MOBILE BANKING REVIEW MINING PIPELINE")
    print("  Steps 1-4: Scrape → Preprocess → Classify → Sentiment")
    print("█" * 60 + "\n")

    # Step 1: Scrape
    if skip_scraping:
        raw_path = os.path.join(OUTPUT_DIR, "step1_raw_reviews.csv")
        if os.path.exists(raw_path):
            print("[*] Loading existing raw data...")
            df = pd.read_csv(raw_path)
        else:
            print("[!] No existing data found. Running scraper...")
            df = step1_scrape_reviews()
    else:
        df = step1_scrape_reviews()

    # Step 2: Preprocess
    df = step2_preprocess(df)

    # Step 3: Classify
    df = step3_classify(df)

    # Step 4: Sentiment
    df = step4_sentiment(df)

    # Final summary
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE!")
    print("█" * 60)
    print(f"\nOutput files in '{OUTPUT_DIR}/':")
    print(f"  1. step1_raw_reviews.csv       — Raw scraped data")
    print(f"  2. step2_preprocessed.csv       — Cleaned & filtered")
    print(f"  3. step3_classified.csv         — Theme-tagged")
    print(f"  4. step4_sentiment_final.csv    — Final with sentiment")
    print(f"  5. pipeline_summary.json        — Summary statistics")
    print(f"\nReady for Step 5 (Statistical Analysis) and Step 6 (Qualitative Coding)")

    return df


# =============================================================
# RUN
# =============================================================
if __name__ == "__main__":
    # First time: run full pipeline
    df = run_pipeline(skip_scraping=False)

    # If you already scraped and want to re-run preprocessing:
    # df = run_pipeline(skip_scraping=True)
