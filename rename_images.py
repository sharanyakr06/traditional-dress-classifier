import os

folders = {
    "kerala": "KL",
    "punjab": "PB",
    "rajasthan": "RJ"
}

for folder, prefix in folders.items():

    files = os.listdir(folder)

    image_files = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    image_files.sort()

    for i, old_name in enumerate(image_files, start=1):

        extension = os.path.splitext(old_name)[1]

        new_name = f"{prefix}{i}{extension}"

        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

    print(f"{folder}: {len(image_files)} images renamed")

print("All images renamed successfully!")