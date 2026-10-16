"""Audit awal dataset penyakit daun mangga di dalam file ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASSES = {
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
}


def get_class_name(file_name: str) -> str:
    """Mengambil nama kelas dari folder induk langsung sebuah gambar."""
    return PurePosixPath(file_name).parent.name


def calculate_difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Membuat dHash sederhana untuk menandai gambar yang tampak sangat mirip."""
    grayscale = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(grayscale.getdata())
    bits = []

    for row in range(hash_size):
        start = row * (hash_size + 1)
        for column in range(hash_size):
            bits.append(pixels[start + column] > pixels[start + column + 1])

    value = sum(bit << index for index, bit in enumerate(bits))
    return f"{value:0{hash_size * hash_size // 4}x}"


def audit_zip(zip_path: Path) -> dict:
    """Memeriksa struktur, isi gambar, dan duplikat dalam sebuah ZIP."""
    class_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    visual_hashes: dict[str, list[str]] = defaultdict(list)
    damaged_images: list[dict[str, str]] = []
    image_sizes: Counter[str] = Counter()

    with zipfile.ZipFile(zip_path) as archive:
        bad_crc_file = archive.testzip()
        image_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and PurePosixPath(member.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]

        for index, member in enumerate(image_members, start=1):
            extension = PurePosixPath(member.filename).suffix.lower()
            class_name = get_class_name(member.filename)
            extension_counts[extension] += 1
            class_counts[class_name] += 1

            try:
                image_bytes = archive.read(member)
                exact_hash = hashlib.sha256(image_bytes).hexdigest()
                exact_hashes[exact_hash].append(member.filename)

                # load() benar-benar mendekode piksel sehingga kerusakan gambar terdeteksi.
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    image_sizes[f"{image.width}x{image.height}"] += 1
                    visual_hash = calculate_difference_hash(image)
                    visual_hashes[visual_hash].append(member.filename)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                damaged_images.append(
                    {"file": member.filename, "error": str(error)}
                )

            if index % 500 == 0:
                print(f"Memeriksa gambar {index}/{len(image_members)}...")

    exact_duplicate_groups = [
        files for files in exact_hashes.values() if len(files) > 1
    ]
    visual_similarity_groups = [
        files for files in visual_hashes.values() if len(files) > 1
    ]

    return {
        "zip_file": str(zip_path.resolve()),
        "zip_size_bytes": zip_path.stat().st_size,
        "bad_crc_file": bad_crc_file,
        "total_images": len(image_members),
        "class_counts": dict(sorted(class_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "image_size_counts": dict(image_sizes.most_common()),
        "damaged_image_count": len(damaged_images),
        "damaged_images": damaged_images,
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_groups": exact_duplicate_groups,
        "same_visual_hash_group_count": len(visual_similarity_groups),
        "same_visual_hash_groups": visual_similarity_groups,
        "expected_classes_found": sorted(EXPECTED_CLASSES & set(class_counts)),
        "missing_expected_classes": sorted(EXPECTED_CLASSES - set(class_counts)),
        "unexpected_classes": sorted(set(class_counts) - EXPECTED_CLASSES),
        "limitations": [
            "Hash SHA-256 hanya menemukan file dengan isi biner identik.",
            "dHash hanya memberi indikasi kemiripan visual dan dapat menghasilkan false positive.",
            "Tanpa ID daun atau metadata pengambilan gambar, varian dari daun yang sama "
            "tidak dapat diidentifikasi secara pasti.",
        ],
    }


def print_summary(result: dict) -> None:
    """Menampilkan ringkasan audit yang mudah dibaca."""
    print("\n=== RINGKASAN AUDIT DATASET ===")
    print(f"Jumlah gambar       : {result['total_images']}")
    print(f"Gambar rusak        : {result['damaged_image_count']}")
    print(f"Grup duplikat persis: {result['exact_duplicate_group_count']}")
    print(f"Grup hash visual sama: {result['same_visual_hash_group_count']}")
    print(f"File dengan CRC rusak: {result['bad_crc_file'] or 'Tidak ada'}")

    print("\nJumlah per kelas:")
    for class_name, count in result["class_counts"].items():
        print(f"- {class_name}: {count}")

    print("\nFormat file:")
    for extension, count in result["extension_counts"].items():
        print(f"- {extension}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset gambar dalam ZIP.")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("archive.zip"),
        help="Lokasi ZIP dataset (default: archive.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_audit.json"),
        help="Lokasi laporan JSON",
    )
    args = parser.parse_args()

    if not args.zip_path.is_file():
        raise SystemExit(
            f"Dataset tidak ditemukan: {args.zip_path.resolve()}\n"
            "Letakkan archive.zip di folder project atau gunakan --zip LOKASI_FILE."
        )

    try:
        result = audit_zip(args.zip_path)
    except zipfile.BadZipFile as error:
        raise SystemExit(f"File bukan ZIP yang valid: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(result)
    print(f"\nLaporan lengkap disimpan di: {args.output.resolve()}")


if __name__ == "__main__":
    main()
