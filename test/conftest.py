import os
import sys

# make `import src.*` work when pytest is invoked from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
