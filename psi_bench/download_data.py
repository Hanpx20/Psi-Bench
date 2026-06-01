#!/usr/bin/env python
"""
Data download script for psi_bench
Downloads datasets from GitHub or Hugging Face
"""

import os
import sys
import json
import argparse
from pathlib import Path
import urllib.request
import tarfile
import zipfile

def download_file(url, output_path, chunk_size=8192):
    """Download file with progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading from: {url}")
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

        print(f"\n✅ Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """Extract tar.gz or zip file"""
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        if str(archive_path).endswith('.tar.gz'):
            print(f"Extracting tar.gz file...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif str(archive_path).endswith('.zip'):
            print(f"Extracting zip file...")
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_to)
        else:
            print("❌ Unsupported archive format")
            return False

        print(f"✅ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def download_from_github(dataset_name, output_dir="./data"):
    """
    Download dataset from GitHub releases
    Requires: psi_bench releases with data attached
    """
    base_url = "https://github.com/Hanpx20/psi_bench/releases/download"

    datasets = {
        "cmv": f"{base_url}/v0.1.0/psi_bench-data-cmv.tar.gz",
        "counsel": f"{base_url}/v0.1.0/psi_bench-data-counsel.tar.gz",
        "request": f"{base_url}/v0.1.0/psi_bench-data-request.tar.gz",
        "all": f"{base_url}/v0.1.0/psi_bench-data-all.tar.gz",
    }

    if dataset_name not in datasets:
        print(f"Available datasets: {', '.join(datasets.keys())}")
        return False

    url = datasets[dataset_name]
    archive_path = Path(output_dir) / f"psi_bench-data-{dataset_name}.tar.gz"

    # Download
    if not download_file(url, archive_path):
        return False

    # Extract
    if not extract_archive(archive_path, output_dir):
        return False

    # Cleanup
    archive_path.unlink()
    print(f"✅ {dataset_name} dataset ready in {output_dir}/")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download psi_bench datasets from GitHub"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["cmv", "counsel", "request", "all"],
        help="Dataset to download (or 'all' for complete dataset)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data",
        help="Output directory (default: ./data)"
    )

    args = parser.parse_args()

    if not args.dataset:
        parser.print_help()
        print("\n💡 Hint: Run 'psi_bench download all' to get all datasets")
        return 1

    print(f"📥 psi_bench Data Download (from GitHub Hanpx20/psi_bench)\n")

    success = download_from_github(args.dataset, args.output)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
