import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

input_file = "data/BACE1_cleaned.csv"
train_file = "data/train.csv"
test_file = "data/test.csv"

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split 80/20
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Saving train set ({len(train_df)} samples) to {train_file}...")
train_df.to_csv(train_file, index=False)

print(f"Saving test set ({len(test_df)} samples) to {test_file}...")
test_df.to_csv(test_file, index=False)

print("Done.")
