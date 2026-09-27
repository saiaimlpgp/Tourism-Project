
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the repository data folder
data_path = "tourism_project/data/tourism.csv"
df = pd.read_csv(data_path)

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Split the cleaned dataset into training and testing sets
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ProdTaken"]
)

# Save the train and test datasets locally
train_path = "tourism_project/model_building/train.csv"
test_path = "tourism_project/model_building/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Data preparation completed successfully.")
print("Cleaned dataset shape:", df.shape)
print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)
print("Training file:", train_path)
print("Testing file:", test_path)
