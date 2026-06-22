import os
import hashlib
import requests
import json
from pathlib import Path

# Official Dataset URLs from the LongMemEval repository README
DATASET_URLS = {
    "longmemeval_oracle.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "longmemeval_s_cleaned.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    # "longmemeval_m_cleaned.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json"
}

def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to avoid high memory usage for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_datasets(output_dir: str = "cognicore_benchmarks/data/longmemeval"):
    """Downloads official datasets and computes metadata hashes."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "repository": "https://github.com/xiaowu0162/LongMemEval.git",
        "benchmark_commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc", # Pinned to repo audit state
        "datasets": {}
    }
    
    for filename, url in DATASET_URLS.items():
        filepath = out_path / filename
        print(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            sha256 = compute_sha256(filepath)
            print(f"  [OK] Downloaded. SHA-256: {sha256}")
            
            metadata["datasets"][filename] = {
                "url": url,
                "sha256": sha256
            }
        except Exception as e:
            print(f"  [x] Failed to download {filename}: {e}")
            
    # Save verification metadata
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nMetadata saved to {out_path / 'metadata.json'}")

if __name__ == "__main__":
    download_datasets()
