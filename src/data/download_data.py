"""Download Amazon Reviews 2023 data from Hugging Face."""

import os
import yaml
import pandas as pd
from huggingface_hub import hf_hub_download


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_reviews(cfg):
    """Download review data for the specified category via hf_hub_download."""
    category = cfg["data"]["category"]
    dataset_name = cfg["data"]["dataset_name"]
    max_samples = cfg["data"]["max_samples"]

    print(f"Loading reviews for category: {category}")
    print(f"Dataset: {dataset_name}")

    local_path = hf_hub_download(
        repo_id=dataset_name,
        filename=f"raw/review_categories/{category}.jsonl",
        repo_type="dataset",
    )

    df = pd.read_json(local_path, lines=True)
    print(f"Loaded {len(df)} reviews")

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=cfg["project"]["seed"])
        print(f"Subsampled to {len(df)} reviews")

    return df


def download_metadata(cfg):
    """Download item metadata for the specified category."""
    category = cfg["data"]["category"]
    dataset_name = cfg["data"]["dataset_name"]

    print(f"Loading metadata for category: {category}")

    try:
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=f"raw/meta_categories/meta_{category}.jsonl",
            repo_type="dataset",
        )
        meta_df = pd.read_json(local_path, lines=True)
        print(f"Loaded metadata for {len(meta_df)} products")
        return meta_df
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None


def save_raw_data(reviews_df, meta_df, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)

    reviews_path = os.path.join(output_dir, "reviews.parquet")
    reviews_df.to_parquet(reviews_path, index=False)
    print(f"Saved reviews to {reviews_path}")

    if meta_df is not None:
        meta_path = os.path.join(output_dir, "metadata.parquet")
        meta_df.to_parquet(meta_path, index=False)
        print(f"Saved metadata to {meta_path}")


def main(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    reviews_df = download_reviews(cfg)
    meta_df = download_metadata(cfg)
    save_raw_data(reviews_df, meta_df)
    print("Download complete.")
    return reviews_df, meta_df


if __name__ == "__main__":
    main()
