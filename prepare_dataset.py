import os
import shutil
import random
from pathlib import Path

# =========================
# SETTINGS
# =========================

BASE_DIR = Path(".")

ANIMALS_DIR = BASE_DIR / "animals"
CARS_DIR = BASE_DIR / "cars"
FOG_DIR = BASE_DIR / "Fog"

OUTPUT_DIR = BASE_DIR / "dataset"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

CLASSES = ["safe", "accident", "fog", "animal"]
SPLITS = ["train", "val", "test"]

random.seed(42)


# =========================
# HELPER FUNCTIONS
# =========================

def create_folders():
    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def is_image(file):
    return file.suffix.lower() in IMAGE_EXTENSIONS


def copy_images(files, destination, prefix):
    destination.mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(files):
        if is_image(file):
            new_name = f"{prefix}_{i}{file.suffix.lower()}"
            shutil.copy2(file, destination / new_name)


def get_all_images(folder):
    return [f for f in folder.rglob("*") if f.is_file() and is_image(f)]


# =========================
# 1. COPY ANIMALS DATA
# =========================

def process_animals():
    print("Processing animals...")

    for split in SPLITS:
        source = ANIMALS_DIR / split
        destination = OUTPUT_DIR / split / "animal"

        if source.exists():
            files = get_all_images(source)
            copy_images(files, destination, f"animal_{split}")

    print("Animals done.")


# =========================
# 2. COPY CARS DATA
# =========================

def process_cars():
    print("Processing cars...")

    for split in SPLITS:
        split_dir = CARS_DIR / split

        if not split_dir.exists():
            continue

        for folder in split_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name.lower()

            if "accident" in folder_name and "non" not in folder_name:
                target_class = "accident"

            elif "non" in folder_name or "normal" in folder_name or "safe" in folder_name:
                target_class = "safe"

            else:
                print(f"Skipped unknown folder: {folder}")
                continue

            files = get_all_images(folder)
            destination = OUTPUT_DIR / split / target_class
            copy_images(files, destination, f"{target_class}_{split}")

    print("Cars done.")


# =========================
# 3. PROCESS FOG DATA
# =========================

def process_fog():
    print("Processing fog...")

    files = get_all_images(FOG_DIR)
    random.shuffle(files)

    total = len(files)

    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    copy_images(train_files, OUTPUT_DIR / "train" / "fog", "fog_train")
    copy_images(val_files, OUTPUT_DIR / "val" / "fog", "fog_val")
    copy_images(test_files, OUTPUT_DIR / "test" / "fog", "fog_test")

    print("Fog done.")


# =========================
# 4. SHOW FINAL SUMMARY
# =========================

def show_summary():
    print("\nDATASET SUMMARY")
    print("=" * 40)

    for split in SPLITS:
        print(f"\n{split.upper()}")
        for cls in CLASSES:
            folder = OUTPUT_DIR / split / cls
            count = len(get_all_images(folder))
            print(f"{cls}: {count} images")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    create_folders()
    process_animals()
    process_cars()
    process_fog()
    show_summary()

    print("\nDataset preparation completed successfully.")