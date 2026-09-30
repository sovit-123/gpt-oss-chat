"""Checks that semantic_engine stays cheap to import.

The embedding model used to load at import time, which made every import
of the module slow and network-dependent. The check runs in a fresh
subprocess so no other test in the suite can pollute it.
"""

import subprocess
import sys


def test_import_does_not_load_the_embedding_model():
    code = (
        "import semantic_engine, sys; "
        "assert semantic_engine._encoder is None; "
        "assert 'sentence_transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
