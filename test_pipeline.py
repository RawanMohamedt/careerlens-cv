"""Quick checks for cv_pipeline.py.

Run:  python test_pipeline.py                 (logic tests only)
      python test_pipeline.py path/to/cv.pdf  (+ end-to-end test, needs models/)
"""
import sys
from pathlib import Path

import cv_pipeline as cp

REF_YM = 2019 * 12 + 6  # June 2019

SAMPLE = """HR MANAGER
Summary
Experienced HR professional.
Skills
Recruiting, payroll, employee relations
Experience
HR Manager Jan 2012 to Current
HR Assistant 03/2008 - 12/2011
Education
BA Business 2007
"""


def test_sections():
    title, secs = cp.split_sections(SAMPLE)
    assert title == "HR MANAGER"
    assert {"summary", "skills", "experience", "education"} <= set(secs)


def test_experience_years():
    _, secs = cp.split_sections(SAMPLE)
    exp = cp.experience_features(secs["experience"], REF_YM)
    assert exp["jobs_found"] == 2
    assert 11 <= exp["years"] <= 12        # 03/2008 -> 06/2019, no double counting


def test_overlap_not_double_counted():
    a = (2010 * 12 + 1, 2012 * 12 + 1)
    b = (2011 * 12 + 1, 2013 * 12 + 1)
    assert cp.total_months([a, b]) == 3 * 12


def test_levels():
    assert cp.assign_level("Intern", 0.5)[0] == "Junior"
    assert cp.assign_level("Accountant", 3)[0] == "Mid"
    assert cp.assign_level("Senior Engineer", 6)[0] == "Senior"
    assert cp.assign_level("Accountant", None)[0] == "Unknown"


def test_clean_text_removes_contact_info():
    out = cp.clean_text("mail me at a.b@x.com or +1 555-123-4567 www.site.com")
    assert "@" not in out and "555" not in out and "www" not in out


def test_end_to_end(pdf_path):
    analyzer = cp.CVAnalyzer()
    out = analyzer.analyze_pdf(pdf_path)
    assert len(out["categories"]) == 3
    assert "confident" in out and "level" in out
    print(out["categories"], out["level"], out["experience"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and k != "test_end_to_end"]
    for t in tests:
        t()
        print("OK ", t.__name__)
    if len(sys.argv) > 1:
        test_end_to_end(sys.argv[1])
        print("OK  test_end_to_end")
    print("All checks passed.")
