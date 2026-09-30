"""Checks that importing semantic_engine stays cheap: the embedding
model must not load at import time. Runs in a subprocess so no other
test can pollute the check."""

import subprocess
import sys


def test_import_does_not_load_the_embedding_model():
    code = (
        "import semantic_engine, sys; "
        "assert semantic_engine._encoder is None; "
        "assert 'sentence_transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
