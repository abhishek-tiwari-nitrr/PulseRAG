"""Start the dashboard: uv run streamlit run streamlit_app.py."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pulserag.ui.app import main

main()
