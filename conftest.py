import sys
from pathlib import Path

# Add the repo root to sys.path so that modules at the root level can be imported.
# Use append (not insert) to avoid shadowing stdlib/packages with same names.
root = str(Path(__file__).parent.resolve())
if root not in sys.path:
    sys.path.append(root)
