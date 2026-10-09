
from pathlib import Path
import hashlib

BASE = Path(r"D:\projects\agribrain\Datasets\2")
FOLDERS = ["original", "processed"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


hashes = {}

for folder_name in FOLDERS:
    root = BASE / folder_name
    current = {}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            current[sha256(path)] = path

    hashes[folder_name] = current
    print(f"{folder_name}: {len(current)} unique file hashes")

original_hashes = set(hashes["original"])
processed_hashes = set(hashes["processed"])

print("\nExact matching files:", len(original_hashes & processed_hashes))
print("Only in original:", len(original_hashes - processed_hashes))
print("Only in processed:", len(processed_hashes - original_hashes))
