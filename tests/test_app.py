from streamlit.testing.v1 import AppTest
from pathlib import Path
def test_app_executes():
    app=AppTest.from_file(Path(__file__).parents[1]/"app.py",default_timeout=120).run()
    assert not app.exception

