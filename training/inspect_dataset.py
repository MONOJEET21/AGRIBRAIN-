
from pathlib import Path

DATASET_DIR = Path(r"D:\projects\agribrain\Datasets\2")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("Dataset:", DATASET_DIR)
print("Exists:", DATASET_DIR.exists())

if not DATASET_DIR.exists():
    raise SystemExit("Dataset path not found. Check the path.")

for section in sorted(DATASET_DIR.iterdir()):
    if not section.is_dir():
        continue

    print(f"\n{'=' * 55}")
    print(f"SECTION: {section.name}")
    print("=" * 55)

    total = 0

    for folder in sorted(section.rglob("*")):
        if folder.is_dir():
            count = sum(
                1 for f in folder.iterdir()
                if f.is_file()
                and f.suffix.lower() in IMAGE_EXTENSIONS
            )

            if count:
                print(f"{folder.relative_to(section)}: {count} images")
                total += count

    print("Total images:", total)
