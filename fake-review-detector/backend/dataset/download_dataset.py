# download_dataset.py

import os
from datasets import load_dataset

def download_amazon_dataset(save_path="backend/dataset/data"):
    print("📦 Downloading Amazon Polarity dataset...")
    # Download the dataset
    dataset = load_dataset("mteb/amazon_polarity", split="train")

    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "amazon_polarity.csv")

    print(f"💾 Saving to {output_file} ...")
    # Convert to pandas DataFrame and save as CSV
    df = dataset.to_pandas()
    df.to_csv(output_file, index=False)
    print("✅ Dataset downloaded and saved.")

if __name__ == "__main__":
    download_amazon_dataset()
