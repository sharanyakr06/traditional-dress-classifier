import os
import csv

folders = {
    "kerala": "Kerala",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan"
}

with open("dataset.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    # CSV headings
    writer.writerow(["image_name", "state"])

    for folder, state in folders.items():

        files = os.listdir(folder)

        image_files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

        # Sort images numerically: KL1, KL2, KL3 ... KL100
        image_files.sort(
            key=lambda x: int(''.join(filter(str.isdigit, x)))
        )

        for image_name in image_files:
            writer.writerow([image_name, state])

print("dataset.csv created successfully!")