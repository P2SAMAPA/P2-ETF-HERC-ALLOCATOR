"""
Upload results to Hugging Face dataset repository.
"""

import json
import os
from huggingface_hub import HfApi, upload_file

import config


def push_daily_result(payload: dict):
    """
    Saves payload as JSON and uploads to HF dataset repo.
    Filename format: herc_weights_YYYY-MM-DD.json
    """
    filename = f"herc_weights_{config.TODAY}.json"

    # Save locally
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved local file: {filename}")

    # Upload to Hugging Face
    if config.HF_TOKEN:
        api = HfApi(token=config.HF_TOKEN)
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=config.HF_OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.HF_OUTPUT_REPO}/{filename}")
    else:
        print("HF_TOKEN not set. Skipping upload.")


def push_file(local_path: str, path_in_repo: str):
    """
    Uploads an arbitrary local file to the HF results dataset at the given
    path_in_repo (e.g. 'backtests/COMBINED_monthly_20260826_101500/comparison.csv').
    Used for pushing backtest artifacts (CSV/PNG/JSON) alongside the daily
    herc_weights_*.json files, without overwriting them.
    """
    if config.HF_TOKEN:
        api = HfApi(token=config.HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=config.HF_OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.HF_OUTPUT_REPO}/{path_in_repo}")
    else:
        print(f"HF_TOKEN not set. Skipping upload of {local_path}.")
