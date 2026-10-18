"""
cv_pipeline.py - CareerLens CV analysis pipeline.

CV (PDF or text)  ->  structured profile (a plain dict that can be returned as JSON).
Steps: extract text -> split sections -> skills -> years of experience -> level -> category.
"""
import datetime as dt
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "category_model.joblib"
DEFAULT_VOCAB = BASE_DIR / "models" / "skills_vocab.json"
MIN_WORDS = 50          # below this the text is almost certainly unreadable (e.g. a scanned PDF)
MIN_CONFIDENCE = 0.4    # below this the top category is a weak guess; let the user pick the track

# ---------------------------------------------------------------- 1) text
EMAIL_RE = re.compile(r"[\w\.\-+]+@[\w\.\-]+\.\w+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
PHONE_RE = re.compile(r"(?<!\w)(\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\w)")
LOC_RE = re.compile(r"\bCity\s*,\s*State\b")          # placeholders of the training dataset
COMP_RE = re.compile(r"Company Name\s*[－—–\-]*")      # (harmless on real CVs)
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


GLUED_RE = re.compile(r"[A-Za-z]{18,}")


def looks_glued(text):
    """True when words are stuck together (e.g. 'Trainedamachinelearningmodel'),
    which happens with some PDF fonts under pdfplumber's default spacing."""
    tokens = text.split()
    if len(tokens) < 20:
        return False
    return sum(bool(GLUED_RE.search(t)) for t in tokens) / len(tokens) > 0.05


def read_pdf(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        if looks_glued(text):  # retry with tighter word spacing
            text = "\n".join((page.extract_text(x_tolerance=1) or "") for page in pdf.pages)
        return text


def clean_text(text):
    text = str(text)
    for rx in (URL_RE, EMAIL_RE, PHONE_RE, LOC_RE, COMP_RE, CTRL_RE):
        text = rx.sub(" ", text)
    text = text.replace("－", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def extract_cv_text(pdf_path):
    from ftfy import fix_text
    return clean_text(fix_text(read_pdf(pdf_path)))


# ------------------------------------------------------------ 2) sections
SECTION_PATTERNS = {
    "summary": r"(professional |career |executive )?(summary|profile|objective)|career objective|summary of qualifications",
    "experience": r"((professional|work|relevant|related) )?(experience|history)|employment( history)?",
    "education": r"education( (and|&) (training|certifications?))?|(academic|educational) (background|qualifications)",
    "skills": r"((technical|key|core|computer) )?skills|skill highlights|highlights|(core )?(qualifications|competencies)|areas of expertise",
    "accomplishments": r"accomplishments|achievements|awards( and honors)?|honors|activities and honors",
    "additional": r"additional information|interests|affiliations|languages|certifications?|references",
}
SECTION_RES = {k: re.compile(rf"^(?:{v})$", re.I) for k, v in SECTION_PATTERNS.items()}


def detect_heading(line):
    s = re.sub(r"[^a-z& ]", " ", line.lower())
    s = re.sub(r"\s+", " ", s).strip()
    if not s or len(s.split()) > 5:
        return None
    for name, rx in SECTION_RES.items():
        if rx.match(s):
            return name
    return None


def split_sections(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title = ""
    if lines and detect_heading(lines[0]) is None:
        title = lines[0]
        lines = lines[1:]
    sections, current = {}, "header"
    for line in lines:
        heading = detect_heading(line)
        if heading:
            current = heading
            continue
        sections.setdefault(current, []).append(line)
    return title, {k: "\n".join(v) for k, v in sections.items()}


# --------------------------------------------------------------- 3) skills
def ngram_set(text, n_max=4):
    tokens = [t.strip(".-/&") for t in re.findall(r"[a-z0-9][a-z0-9+#./&-]*", text.lower())]
    tokens = [t for t in tokens if t]
    grams = set()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            grams.add(" ".join(tokens[i:i + n]))
    return grams


def extract_skills(skills_section, body, vocab):
    listed = ngram_set(skills_section) & vocab
    elsewhere = (ngram_set(body) & vocab) - listed
    return sorted(listed), sorted(elsewhere)


# ----------------------------------------------------------- 4) experience
MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def date_pattern(p):
    return (rf"(?:(?P<{p}m>{MONTH})\s*,?\s*(?P<{p}y1>(?:19|20)\d{{2}})"
            rf"|(?P<{p}n>\d{{1,2}})\s*/\s*(?P<{p}y2>(?:19|20)\d{{2}})"
            rf"|(?P<{p}y3>(?:19|20)\d{{2}}))")


RANGE_RE = re.compile(
    rf"(?<![\w/]){date_pattern('s')}\s*(?:to|until|through|-|–|—)\s*"
    rf"(?:{date_pattern('e')}|(?P<cur>current|present|now|today|till date|to date|ongoing))",
    re.IGNORECASE,
)


def parse_date(p, g):
    year = g.get(p + "y1") or g.get(p + "y2") or g.get(p + "y3")
    if not year:
        return None
    if g.get(p + "m"):
        month = MONTHS[g[p + "m"][:3].lower()]
    elif g.get(p + "n"):
        month = int(g[p + "n"])
        if not 1 <= month <= 12:
            return None
    else:
        month = 6
    return int(year) * 12 + month


def extract_ranges(text, ref_ym):
    ranges = []
    for m in RANGE_RE.finditer(text):
        g = m.groupdict()
        start = parse_date("s", g)
        end = ref_ym if g.get("cur") else parse_date("e", g)
        if start is None or end is None:
            continue
        if start > ref_ym or end < start or end - start > 12 * 50:
            continue
        ranges.append((start, min(end, ref_ym)))
    return ranges


def total_months(ranges):
    if not ranges:
        return 0
    ranges = sorted(ranges)
    cur_s, cur_e = ranges[0]
    total = 0
    for s, e in ranges[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def experience_features(experience_section, ref_ym):
    ranges = extract_ranges(experience_section, ref_ym)
    if not ranges:
        return {"years": None, "jobs_found": 0, "first_job_year": None}
    return {
        "years": round(total_months(ranges) / 12, 1),
        "jobs_found": len(ranges),
        "first_job_year": int((min(s for s, _ in ranges) - 1) // 12),
    }


# ------------------------------------------------------------------ 5) level
SENIOR_KW = re.compile(
    r"\b(senior|sr|lead|principal|head|director|vp|vice president|chief|president|ceo|cto|cfo|coo)\b", re.I
)
JUNIOR_KW = re.compile(
    r"\b(intern|internship|trainee|junior|jr|entry[- ]level|apprentice|graduate|student)\b", re.I
)


def years_level(y):
    if y < 2:
        return "Junior"
    if y < 5:
        return "Mid"
    return "Senior"


def assign_level(title, years):
    if years is None:
        if SENIOR_KW.search(title):
            return "Senior", "title (no dates found)"
        if JUNIOR_KW.search(title):
            return "Junior", "title (no dates found)"
        return "Unknown", "no dates and no clear title keyword"
    if JUNIOR_KW.search(title) and years < 4:
        return "Junior", "title + years"
    if SENIOR_KW.search(title) and years >= 3:
        return "Senior", "title + years"
    return years_level(years), "years"


# ---------------------------------------------------------------- 5-B) tracks
# Rule-based track detection (no training data needed).
# To add or change a track, edit this dictionary only:
#   "title"    : phrases that count strongly when they appear in the first 3 lines (the headline)
#   "keywords" : skills / terms of the track (all lowercase)
# Matching weights: headline = 3, inside the Skills section = 2, anywhere else = 1.
TRACKS = {
    "Backend Development": {
        "title": ["backend", "back end", "back-end", "server side", "net developer",
                  "java developer", "python developer", "node.js developer", "api developer"],
        "keywords": ["c#", "asp.net", "asp.net core", "entity framework", "linq", "sql server",
                     "java", "spring boot", "node.js", "express.js", "django", "flask", "fastapi",
                     "php", "laravel", "rest api", "rest apis", "web api", "microservices",
                     "postgresql", "mysql", "mongodb", "sql", "jwt"],
    },
    "Frontend Development": {
        "title": ["frontend", "front end", "front-end", "react developer", "angular developer",
                  "ui developer"],
        "keywords": ["html", "html5", "css", "css3", "javascript", "typescript", "react",
                     "angular", "vue", "bootstrap", "tailwind", "sass", "responsive design",
                     "redux", "next.js", "jquery"],
    },
    "Full Stack Web Development": {
        "title": ["full stack", "fullstack", "full-stack", "web developer"],
        "keywords": ["full stack", "mern", "web application", "web applications",
                     "web development", "asp.net mvc", "mvc", "html", "css", "javascript",
                     "sql", "rest api", "asp.net", "node.js", "react"],
    },
    "Data Science / Machine Learning": {
        "title": ["data scientist", "machine learning", "ml engineer", "ai engineer",
                  "deep learning"],
        "keywords": ["python", "machine learning", "deep learning", "scikit-learn", "sklearn",
                     "tensorflow", "keras", "pytorch", "pandas", "numpy", "matplotlib",
                     "seaborn", "statistics", "probability", "linear algebra",
                     "neural networks", "nlp", "computer vision", "tf-idf", "classification",
                     "regression", "feature engineering", "xgboost", "jupyter"],
    },
    "Data Analysis / BI": {
        "title": ["data analyst", "business intelligence", "bi developer", "bi analyst"],
        "keywords": ["excel", "power bi", "tableau", "sql", "data visualization",
                     "data analysis", "dashboards", "pandas", "etl", "statistics", "looker",
                     "google analytics", "data cleaning"],
    },
    "Mobile Development": {
        "title": ["mobile developer", "android developer", "ios developer", "flutter developer",
                  "react native"],
        "keywords": ["android", "ios", "kotlin", "swift", "flutter", "dart", "react native",
                     "xamarin", "firebase", "mobile app", "mobile apps"],
    },
    "DevOps / Cloud": {
        "title": ["devops", "cloud engineer", "site reliability"],
        "keywords": ["docker", "kubernetes", "aws", "azure", "ci/cd", "jenkins", "linux",
                     "terraform", "ansible", "github actions"],
    },
}

MIN_TRACK_SCORE = 4     # below this there is not enough evidence for any track
MIN_TRACK_SHARE = 0.40  # the top track must hold at least this share of all scores
MIN_TRACK_MARGIN = 0.10 # ...and lead the second track by at least this much (else: ambiguous)


def _norm_kw(kw):
    tokens = [t.strip(".-/&") for t in re.findall(r"[a-z0-9][a-z0-9+#./&-]*", kw.lower())]
    return " ".join(t for t in tokens if t)


def detect_tracks(header_text, skills_section, full_text, tracks=None, top_k=3,
                  min_score=MIN_TRACK_SCORE, min_share=MIN_TRACK_SHARE,
                  min_margin=MIN_TRACK_MARGIN):
    tracks = tracks or TRACKS
    head, listed, whole = ngram_set(header_text), ngram_set(skills_section), ngram_set(full_text)

    rows = []
    for name, cfg in tracks.items():
        score, matched = 0, []
        for kw in {_norm_kw(k) for k in cfg.get("title", [])}:
            if kw in head:
                score += 3
                matched.append(kw)
        for kw in {_norm_kw(k) for k in cfg.get("keywords", [])} - set(matched):
            if kw in listed:
                score += 2
                matched.append(kw)
            elif kw in whole:
                score += 1
                matched.append(kw)
        rows.append((name, score, sorted(matched)))

    total = sum(r[1] for r in rows)
    rows.sort(key=lambda r: r[1], reverse=True)
    result = [
        {"track": n, "score": sc, "share": round(sc / total, 3) if total else 0.0,
         "matched": m[:10]}
        for n, sc, m in rows[:top_k] if sc > 0
    ]
    confident = bool(result) and result[0]["score"] >= min_score and result[0]["share"] >= min_share
    if confident and len(result) > 1:
        confident = result[0]["share"] - result[1]["share"] >= min_margin
    return result, confident


# -------------------------------------------------------------- 6) analyzer
class CVAnalyzer:
    def __init__(self, model_path=DEFAULT_MODEL, vocab_path=DEFAULT_VOCAB,
                 min_words=MIN_WORDS, min_confidence=MIN_CONFIDENCE, tracks=None):
        self.tracks = tracks or TRACKS
        self.min_words = min_words
        self.min_confidence = min_confidence
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.vocab = set(json.loads(Path(vocab_path).read_text(encoding="utf-8")))

    def analyze_text(self, text, ref_date=None, top_k=3):
        ref_date = ref_date or dt.date.today()
        ref_ym = ref_date.year * 12 + ref_date.month

        text = clean_text(text)
        if len(text.split()) < self.min_words:
            raise ValueError(
                f"CV text is too short ({len(text.split())} words). "
                "The file may be a scanned image or unreadable."
            )

        title, sections = split_sections(text)
        body = text.split("\n", 1)[1] if title and "\n" in text else text

        listed, elsewhere = extract_skills(sections.get("skills", ""), body, self.vocab)
        exp = experience_features(sections.get("experience", ""), ref_ym)
        level, basis = assign_level(title, exp["years"])

        header_text = "\n".join([l for l in text.split("\n") if l.strip()][:3])
        track_list, track_confident = detect_tracks(
            header_text, sections.get("skills", ""), text, self.tracks, top_k=top_k
        )

        X = pd.DataFrame({
            "title": [title],
            "body": [body],
            "skills": [" ".join(s.replace(" ", "_") for s in listed + elsewhere)],
        })
        proba = self.model.predict_proba(X)[0]
        names = self.label_encoder.inverse_transform(self.model.classes_)
        order = np.argsort(proba)[::-1][:top_k]

        return {
            "title": title,
            "categories": [
                {"category": str(names[i]), "probability": round(float(proba[i]), 3)} for i in order
            ],
            "confident": bool(proba[order[0]] >= self.min_confidence),
            "tracks": track_list,
            "track_confident": track_confident,
            "experience": exp,
            "level": {"value": level, "basis": basis},
            "skills": {"listed": listed, "mentioned_elsewhere": elsewhere},
            "meta": {
                "n_words": len(text.split()),
                "sections_found": sorted(sections),
                "analysis_date": ref_date.isoformat(),
            },
        }

    def analyze_pdf(self, pdf_path, ref_date=None, top_k=3):
        return self.analyze_text(extract_cv_text(pdf_path), ref_date=ref_date, top_k=top_k)


# ------------------------------------------------------------------- 7) CLI
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Analyze a CV (PDF) and print a JSON profile.")
    ap.add_argument("pdf", help="path to the CV PDF")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    ap.add_argument("--ref-date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    ref = dt.date.fromisoformat(args.ref_date) if args.ref_date else None
    analyzer = CVAnalyzer(args.model, args.vocab)
    result = analyzer.analyze_pdf(args.pdf, ref_date=ref, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
