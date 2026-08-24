"""Prompt parsing.

Imported lazily by `vista.config`: `vista.parsing.extractor` clears
CUDA_VISIBLE_DEVICES at import time to pin the spaCy parser to CPU, so it must
not be imported before the generator has initialised CUDA.
"""

from vista.parsing.extractor import (
    extract_numerical_pairs,
    extract_pairs,
    extract_spatial_pairs,
    resolve_token_ids,
)

__all__ = [
    'extract_pairs',
    'extract_spatial_pairs',
    'extract_numerical_pairs',
    'resolve_token_ids',
]
