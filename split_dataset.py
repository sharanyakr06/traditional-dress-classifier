import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original dataset
df = pd.read_csv("dataset.csv")

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["state"],
    random_state=42
)

# Second split: divide the 30% into 15% validation and 15% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["state"],
    random_state=42
)

# Save the three datasets
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Display results
print("Dataset split successfully! ✅")

print("\nTotal images:", len(df))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\nTraining distribution:")
print(train_df["state"].value_counts())

print("\nValidation distribution:")
print(val_df["state"].value_counts())

print("\nTesting distribution:")
print(test_df["state"].value_counts())