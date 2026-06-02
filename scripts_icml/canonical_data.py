"""Build the one-time canonical balanced pool (data + SBERT + disagreement).

Usage:
    python scripts_icml/canonical_data.py --category All_Beauty --pool-size 20000
"""

import argparse

from scripts_icml.icml_common import build_canonical_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty")
    ap.add_argument("--pool-size", type=int, default=20000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = build_canonical_pool(args.category, pool_size=args.pool_size, force=args.force)
    print("\n[done] pool shape:", df.shape)
    print("label balance:\n", df["label"].value_counts())
    print("agreement_status:\n", df["agreement_status"].value_counts())
    print("disagreement_group:\n", df["disagreement_group"].value_counts())


if __name__ == "__main__":
    main()
