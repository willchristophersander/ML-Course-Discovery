import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.course_discovery import SearchResult, SearchResultClassifier
from models.course_page_classifier import CoursePageClassifier
from models.course_search_classifier import CourseSearchClassifier


def test_course_search_feature_detection():
    classifier = CourseSearchClassifier()
    html = """
    <html>
      <body>
        <form>
          <input type="text" placeholder="Search courses" />
          <button>Search</button>
        </form>
        <div>Course Catalog</div>
      </body>
    </html>
    """
    features = classifier.extract_search_features("https://example.edu/course-search", html)
    assert features["has_search_form"]
    assert features["has_search_input"]
    assert features["url_has_search"]
    assert features["has_catalog_text"]


def test_course_page_feature_counts():
    classifier = CoursePageClassifier()
    html = """
    <html>
      <body>
        <h2>MATH 101 - Calculus I</h2>
        <p>Credits: 4</p>
        <p>Prerequisite: MATH 100</p>
        <table><tr><td>Course Description</td></tr></table>
      </body>
    </html>
    """
    features = classifier.extract_course_features(html)
    assert features["course_code_count"] >= 1
    assert features["credit_mentions"] >= 1
    assert features["prereq_mentions"] >= 1


def test_search_result_feature_extraction():
    classifier = SearchResultClassifier()
    result = SearchResult(
        title="Course Search - Example University",
        url="https://example.edu/course-search",
        snippet="Find courses and search available sections",
        domain="example.edu",
        source="google"
    )
    features = classifier._extract_features(result)
    assert features["has_course_in_url"]
    assert features["title_has_course"]
    assert features["snippet_has_search"]
