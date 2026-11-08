import pandas as pd
import os

# Load CSV
df = pd.read_csv("dataset.csv")

print("CSV loaded successfully!")

print("\nColumns:")
print(df.columns.tolist())

print("\nTotal images:")
print(len(df))

print("\nImages in each state:")
print(df["state"].value_counts())

# Check whether images exist
missing = []

for _, row in df.iterrows():

    state_folder = row["state"].lower()
    image_name = row["image_name"]

    image_path = os.path.join(state_folder, image_name)

    if not os.path.exists(image_path):
        missing.append(image_path)

print("\nMissing images:", len(missing))

if missing:
    print("\nMissing image examples:")
    for image in missing[:10]:
        print(image)
else:
    print("\nALL IMAGES FOUND SUCCESSFULLY! ✅")