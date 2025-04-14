import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", os.path.join("abbvisionsystem", "app.py")]
    sys.exit(stcli.main())
