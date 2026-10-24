"""Download the public BGE encoder once; all subsequent inference is local."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tastelayer' / 'src'))
from tastelayer.model.features import load_bge

load_bge(allow_download=True)
print('BGE model cached. TasteNet can now predict offline.')
