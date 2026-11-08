import pandas as pd
import os
import hashlib


# ==========================================
# SETTINGS
# ==========================================

CSV_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv"
]


# ==========================================
# CALCULATE IMAGE HASH
# ==========================================

def get_hash(image_path):

    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    except Exception:
        return None


# ==========================================
# CHECK ALL IMAGES
# ==========================================

all_images = {}

for csv_file in CSV_FILES:

    df = pd.read_csv(csv_file)

    print(f"\nChecking {csv_file}...")

    for _, row in df.iterrows():

        state = row["state"]
        image_name = row["image_name"]

        image_path = os.path.join(
            state.lower(),
            image_name
        )

        image_hash = get_hash(image_path)

        if image_hash is None:
            print("Could not read:", image_path)
            continue

        if image_hash not in all_images:

            all_images[image_hash] = []

        all_images[image_hash].append(
            (csv_file, image_path)
        )


# ==========================================
# FIND DUPLICATES
# ==========================================

duplicates = []

for image_hash, files in all_images.items():

    if len(files) > 1:

        duplicates.append(files)


# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("DUPLICATE IMAGE CHECK")
print("===================================")

if len(duplicates) == 0:

    print("No exact duplicate images found.")

else:

    print("Duplicate groups found:", len(duplicates))

    for i, group in enumerate(duplicates, 1):

        print(f"\nDuplicate Group {i}:")

        for csv_file, image_path in group:

            print(f"  {csv_file} -> {image_path}")