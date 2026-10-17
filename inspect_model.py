
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, UnidentifiedImageError
import hashlib
import json
import zipfile

# ==========================================
# KONFIGURASI
# ==========================================

ZIP_PATH = Path("archive.zip")
DATASET_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")

EXPECTED_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================
# FUNGSI PEMBANTU
# ==========================================

def sha256_file(path):
    """Menghitung hash file untuk mendeteksi file identik."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_dataset_root():
    """Mencari folder yang langsung berisi delapan folder kelas."""
    candidates = []

    if DATASET_DIR.exists():
        candidates.append(DATASET_DIR)
        candidates.extend(
            path for path in DATASET_DIR.rglob("*") if path.is_dir()
        )

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            names = archive.namelist()

        directories = {
            str(Path(name).parent).replace("\\", "/")
            for name in names
            if name and not name.endswith("/")
        }

        for directory in directories:
            root = Path(directory)
            if root.name.lower() in {
                name.lower() for name in EXPECTED_CLASSES
            }:
                candidates.append(root.parent)

    for root in candidates:
        if all(
            (root / class_name).is_dir()
            for class_name in EXPECTED_CLASSES
        ):
            return root

    return None


def inspect_dataset(root):
    class_counts = Counter()
    extensions = Counter()
    dimensions = Counter()
    hash_groups = defaultdict(list)
    damaged_images = []
    all_images = []

    for class_name in EXPECTED_CLASSES:
        class_dir = root / class_name

        if not class_dir.is_dir():
            print(f"[PERINGATAN] Folder kelas tidak ditemukan: {class_dir}")
            continue

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            all_images.append(path)
            class_counts[class_name] += 1
            extensions[path.suffix.lower()] += 1

            try:
                # Memastikan file dapat dibuka dan strukturnya valid.
                with Image.open(path) as image:
                    image.verify()

                # Buka kembali untuk membaca dimensi.
                with Image.open(path) as image:
                    dimensions[image.size] += 1

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                SyntaxError,
            ) as error:
                damaged_images.append({
                    "path": str(path),
                    "error": str(error),
                })
                continue

            # Hash hanya dihitung untuk file gambar yang valid.
            digest = sha256_file(path)
            hash_groups[digest].append(str(path))

    # Grup duplikat: satu grup terdiri dari minimal 2 file identik.
    duplicate_groups = [
        paths for paths in hash_groups.values()
        if len(paths) > 1
    ]

    duplicate_image_count = sum(
        len(paths) for paths in duplicate_groups
    )

    duplicate_extra_count = sum(
        len(paths) - 1 for paths in duplicate_groups
    )

    results = {
        "dataset_root": str(root),
        "total_images": len(all_images),
        "class_counts": dict(class_counts),
        "extensions": dict(extensions),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in dimensions.items()
        },
        "damaged_image_count": len(damaged_images),
        "damaged_images": damaged_images,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_image_count": duplicate_image_count,
        "duplicate_extra_count": duplicate_extra_count,
        "duplicate_groups": duplicate_groups,
    }

    return results


# ==========================================
# JALANKAN PEMERIKSAAN
# ==========================================

def main():
    print("=" * 60)
    print("PEMERIKSAAN DATASET DAUN MANGGA")
    print("=" * 60)

    root = get_dataset_root()

    if root is None:
        print("Folder dataset dengan delapan kelas tidak ditemukan.")
        print("Pastikan struktur foldernya seperti ini:")
        print("data/raw/")
        for name in EXPECTED_CLASSES:
            print(f"    {name}/")
        print("\nJika dataset masih berbentuk ZIP, ekstrak terlebih dahulu")
        print("ke data/raw/ sesuai struktur folder di atas.")
        return

    print(f"\nLokasi dataset: {root.resolve()}")
    results = inspect_dataset(root)

    print("\n--- JUMLAH GAMBAR PER KELAS ---")
    for class_name in EXPECTED_CLASSES:
        count = results["class_counts"].get(class_name, 0)
        print(f"{class_name}: {count}")

    print("\n--- RINGKASAN DATASET ---")
    print(f"Jumlah seluruh gambar: {results['total_images']}")
    print(f"Jumlah gambar rusak: {results['damaged_image_count']}")
    print(f"Jumlah grup duplikat identik: {results['duplicate_group_count']}")
    print(f"Jumlah file dalam grup duplikat: {results['duplicate_image_count']}")
    print(f"Jumlah salinan tambahan: {results['duplicate_extra_count']}")

    print("\n--- FORMAT GAMBAR ---")
    for extension, count in sorted(results["extensions"].items()):
        print(f"{extension}: {count}")

    print("\n--- UKURAN GAMBAR ---")
    for size, count in sorted(results["image_dimensions"].items()):
        print(f"{size}: {count} gambar")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "dataset_check_results.json"

    try:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
    except PermissionError:
        # Folder outputs dapat dimiliki proses lain pada sebagian lingkungan.
        output_path = Path("dataset_check_results.json")
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nHasil lengkap disimpan ke: {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
