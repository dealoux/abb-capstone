import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abbvisionsystem.app import main

if __name__ == "__main__":
    sys.exit(main())
