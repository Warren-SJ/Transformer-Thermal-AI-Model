import os
import shutil

source_dir = "data"
destination_dir = "separated_data"

faulty_dir = os.path.join(destination_dir, "faulty")
normal_dir = os.path.join(destination_dir, "normal")
os.makedirs(faulty_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if "faulty" in root:
            shutil.copy(os.path.join(root, file), os.path.join(faulty_dir, file))
        elif "normal" in root:
            shutil.copy(os.path.join(root, file), os.path.join(normal_dir, file))

print(f"Data has been separated into '{destination_dir}' folder.")
