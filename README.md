# CareerLens: CV Analysis Module

Takes a CV (PDF) and returns a structured profile as JSON:

- the most likely **tracks** (e.g. Backend, Data Science / ML) with the matched keywords
- extracted **skills**
- **years of experience** and a **level** (Junior / Mid / Senior)
- a predicted job **category** (legacy classifier, see [Known limitations](#known-limitations))

This is the first module of CareerLens, an AI interview and skill-gap analysis platform.
Its output feeds the later stages: track selection, interview questions, answer evaluation and the final report.

## How it works

```
PDF -> text extraction (pdfplumber + ftfy) -> cleaning -> section splitting
    -> skills extraction -> years of experience (date ranges) -> level (rules)
    -> track detection (rules) + category (TF-IDF + Logistic Regression)
```

## Install

```bash
pip install -r requirements.txt
```

The trained files must be in `models/`:

- `category_model.joblib`
- `skills_vocab.json`

The model was trained with `scikit-learn==1.6.1`. Use the same version, otherwise loading the `.joblib` file may fail or behave differently.

## Usage

```python
from cv_pipeline import CVAnalyzer

analyzer = CVAnalyzer()                      # loads models/ by default
result = analyzer.analyze_pdf("cv.pdf")
print(result["tracks"], result["track_confident"])
```

Or from the command line:

```bash
python cv_pipeline.py cv.pdf
```

### Output

```json
{
  "title": "first line of the CV",
  "categories": [{"category": "...", "probability": 0.0}],
  "confident": false,
  "tracks": [
    {"track": "Backend Development", "score": 19, "share": 0.404,
     "matched": ["asp.net", "c#", "sql server"]}
  ],
  "track_confident": false,
  "experience": {"years": 0.5, "jobs_found": 1, "first_job_year": 2025},
  "level": {"value": "Junior", "basis": "years"},
  "skills": {"listed": [], "mentioned_elsewhere": []},
  "meta": {"n_words": 240, "sections_found": [], "analysis_date": "YYYY-MM-DD"}
}
```

See `examples/sample_output.json` for the full shape.

Notes:

- `title` is the first line of the CV. On many real CVs this is the person's name, so treat it as personal data and do not log or share it.
- A `ValueError` is raised when the extracted text is under 50 words (usually a scanned or image-only PDF).

## Track detection

Tracks are detected with a rule-based scorer, so **no training data is needed**.
Each track in the `TRACKS` dictionary (inside `cv_pipeline.py`) has headline phrases and keywords.
A match in the CV headline (first 3 lines) scores 3, inside the Skills section 2, anywhere else 1.

The output has the top 3 tracks with their score share and matched keywords.
`track_confident` is `false` when the evidence is weak or when the top two tracks are close
(for example a "Full Stack .NET Developer" matches both Backend and Full Stack).
In that case let the user choose from the top 3. When it is `true`, the top track can be pre-selected.

To add or change a track, edit the `TRACKS` dictionary only:

```python
TRACKS["Cybersecurity"] = {
    "title": ["security analyst", "penetration tester"],
    "keywords": ["owasp", "penetration testing", "siem", "wireshark", "nmap"],
}
```

Keywords must be lowercase.

## Results

Category classifier, 5-fold stratified cross-validation on the training dataset
(model: TF-IDF on title and body as separate fields + Logistic Regression):

| Metric | Value |
|---|---|
| Accuracy | 0.805 ± 0.007 |
| Macro-F1 | 0.741 ± 0.012 |

These numbers come from CVs of the training dataset, where the first line is a job title.
On real CVs that start with a person's name the title field is much less informative,
so expect lower accuracy on the category. Use `track_confident` and let the user confirm the track.

## Data

The category model was trained on the public Resume dataset on Kaggle (24 job categories).
The data is **not** included in this repository. Download it from Kaggle if you want to retrain.
Do not commit real CVs to this repository.

## Known limitations

- The category classifier is weak on short real-world CVs (tested on two real CVs: probabilities were low). Skills, level and experience were extracted correctly. Prefer `tracks` for choosing the user's track.
- Categories come from a US-style dataset (24 categories) and do not include tracks such as .NET or Machine Learning.
- Scanned or image-only PDFs are not supported (no OCR).
- Some PDF fonts are read without spaces between words. This is detected automatically and the text is re-extracted with tighter spacing.
- The skills vocabulary comes from the training data, so generic words such as "email" or "phone" can appear as skills.
- A `PROJECTS` heading is not recognized as a section, so words from the projects part can be counted as skills mentioned elsewhere.
- The first line of the CV is treated as the title; the section and experience rules assume English CVs with standard headings.
- Level detection is rule-based (no labels exist), so it is an estimate. Experience years need dates in the Experience section.

## Tests

```bash
python test_pipeline.py              # logic tests
python test_pipeline.py cv.pdf       # + end-to-end test (needs models/)
```

## Notebook

`notebooks/01_cv_module.ipynb` contains the full data audit, cleaning, experiments and model selection (run on Google Colab).
https://huggingface.co/spaces/RawanMohamedt/careerlens
