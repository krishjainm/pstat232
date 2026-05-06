"""Script 02: Preprocess raw data, create labels and metadata features."""

import sys
sys.path.insert(0, ".")

from src.data.preprocess import preprocess_pipeline
from src.data.create_splits import create_splits

if __name__ == "__main__":
    preprocess_pipeline()
    create_splits()
