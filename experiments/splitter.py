# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description:
# Note:

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('./data/public_data/train/track_b/eng.csv')

# Set the split ratio (e.g., 80% for training, 20% for validation)
train_ratio = 0.8

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, train_size=train_ratio, random_state=42)

# Optional: Save the splits to separate CSV files
train_df.to_csv('eng_train.csv', index=False)
val_df.to_csv('eng_val.csv', index=False)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")