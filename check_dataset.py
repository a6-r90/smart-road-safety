from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")
SPLITS = ["train", "val", "test"]
CLASSES = ["safe", "accident", "fog", "animal"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def is_image(file):
    return file.suffix.lower() in IMAGE_EXTENSIONS

def check_image(file):
    try:
        with Image.open(file) as img:
            img.verify()
        return True
    except Exception:
        return False

print("\nDATASET QUALITY CHECK")
print("=" * 40)

broken_images = []

for split in SPLITS:
    print(f"\n{split.upper()}")
    for cls in CLASSES:
        folder = DATASET_DIR / split / cls
        images = [f for f in folder.rglob("*") if f.is_file() and is_image(f)]

        print(f"{cls}: {len(images)} images")

        for img in images:
            if not check_image(img):
                broken_images.append(img)

print("\nBROKEN IMAGES")
print("=" * 40)

if len(broken_images) == 0:
    print("No broken images found ✅")
else:
    for img in broken_images:
        print(img)