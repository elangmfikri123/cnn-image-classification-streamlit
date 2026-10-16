"""Training dan evaluasi klasifikasi penyakit daun mangga.

Contoh penggunaan:
    .venv\Scripts\python train.py --model baseline
    .venv\Scripts\python train.py --model tune
    .venv\Scripts\python train.py --model mobilenet
    .venv\Scripts\python train.py --model all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Mengekstrak ZIP sambil mencegah path traversal."""
    marker = destination / ".extraction_complete"
    if marker.exists():
        return

    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Path tidak aman di dalam ZIP: {member.filename}")
        archive.extractall(destination)

    marker.write_text("Ekstraksi selesai.\n", encoding="utf-8")


def find_dataset_root(data_directory: Path) -> Path:
    """Menemukan folder yang langsung berisi delapan folder kelas."""
    candidates = [data_directory] + [path for path in data_directory.rglob("*") if path.is_dir()]
    for candidate in candidates:
        class_directories = [path for path in candidate.iterdir() if path.is_dir()]
        image_class_count = sum(
            any(file.suffix.lower() in IMAGE_EXTENSIONS for file in folder.iterdir() if file.is_file())
            for folder in class_directories
        )
        if image_class_count == 8:
            return candidate
    raise FileNotFoundError("Folder yang berisi delapan kelas tidak ditemukan.")


def calculate_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_difference_hash(file_path: Path, hash_size: int = 8) -> str:
    with Image.open(file_path) as image:
        grayscale = image.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(grayscale.getdata())

    bits = []
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for column in range(hash_size):
            bits.append(pixels[start + column] > pixels[start + column + 1])

    value = sum(bit << index for index, bit in enumerate(bits))
    return f"{value:0{hash_size * hash_size // 4}x}"


def normalize_copy_name(file_path: Path) -> str:
    """Menyamakan nama asli dengan nama salinan seperti '- Copy' atau '(1)'."""
    stem = re.sub(r"(?:\s*-\s*Copy)+", "", file_path.stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    return stem.lower()


class UnionFind:
    """Struktur kecil untuk menggabungkan gambar yang harus berada di split sama."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def remove_exact_duplicates(files: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    unique_files = []
    removed_files = []
    first_file_by_hash: dict[str, Path] = {}

    for file_path in sorted(files):
        file_hash = calculate_file_hash(file_path)
        if file_hash in first_file_by_hash:
            removed_files.append(
                {
                    "removed": str(file_path),
                    "kept": str(first_file_by_hash[file_hash]),
                    "sha256": file_hash,
                }
            )
        else:
            first_file_by_hash[file_hash] = file_path
            unique_files.append(file_path)

    return unique_files, removed_files


def create_groups(files: list[Path]) -> list[list[Path]]:
    """Mengelompokkan gambar dengan dHash atau nama salinan yang sama."""
    union_find = UnionFind(len(files))
    first_by_visual_hash: dict[str, int] = {}
    first_by_copy_name: dict[str, int] = {}

    for index, file_path in enumerate(files):
        visual_hash = calculate_difference_hash(file_path)
        copy_name = normalize_copy_name(file_path)

        if visual_hash in first_by_visual_hash:
            union_find.union(index, first_by_visual_hash[visual_hash])
        else:
            first_by_visual_hash[visual_hash] = index

        if copy_name in first_by_copy_name:
            union_find.union(index, first_by_copy_name[copy_name])
        else:
            first_by_copy_name[copy_name] = index

    groups: dict[int, list[Path]] = defaultdict(list)
    for index, file_path in enumerate(files):
        groups[union_find.find(index)].append(file_path)
    return list(groups.values())


def split_class_groups(groups: list[list[Path]], seed: int) -> dict[str, list[Path]]:
    """Membagi kelompok satu kelas sedekat mungkin ke rasio 70/15/15."""
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total = sum(len(group) for group in groups)
    targets = {
        "train": round(total * TRAIN_RATIO),
        "validation": round(total * VALIDATION_RATIO),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    splits: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}

    for group in groups:
        remaining = {
            split_name: targets[split_name] - len(splits[split_name])
            for split_name in splits
        }
        selected_split = max(remaining, key=remaining.get)
        splits[selected_split].extend(group)

    return splits


def prepare_splits(dataset_root: Path, output_directory: Path) -> pd.DataFrame:
    """Membuat split reproducible dan menyimpannya sebagai CSV."""
    split_csv = output_directory / "data_splits.csv"
    duplicate_json = output_directory / "removed_exact_duplicates.json"
    if split_csv.exists():
        return pd.read_csv(split_csv)

    records = []
    removed_duplicates = []
    class_directories = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    for class_index, class_directory in enumerate(class_directories):
        files = [
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        unique_files, removed = remove_exact_duplicates(files)
        removed_duplicates.extend(removed)
        groups = create_groups(unique_files)
        class_splits = split_class_groups(groups, SEED + class_index)

        for split_name, split_files in class_splits.items():
            for file_path in split_files:
                records.append(
                    {
                        "file_path": str(file_path.resolve()),
                        "class_name": class_directory.name,
                        "split": split_name,
                    }
                )

    split_dataframe = pd.DataFrame(records).sort_values(
        ["split", "class_name", "file_path"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    split_dataframe.to_csv(split_csv, index=False)
    duplicate_json.write_text(
        json.dumps(removed_duplicates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return split_dataframe


def print_split_summary(split_dataframe: pd.DataFrame) -> None:
    summary = pd.crosstab(split_dataframe["class_name"], split_dataframe["split"])
    print("\nDistribusi data setelah menghapus duplikat identik:")
    print(summary)
    print(f"\nTotal gambar untuk eksperimen: {len(split_dataframe)}")


def load_image(file_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def create_dataset(
    split_dataframe: pd.DataFrame,
    split_name: str,
    class_names: list[str],
) -> tf.data.Dataset:
    selected = split_dataframe[split_dataframe["split"] == split_name].copy()
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = selected["class_name"].map(class_to_index).to_numpy(dtype=np.int32)
    paths = selected["file_path"].to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split_name == "train":
        dataset = dataset.shuffle(len(selected), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def build_baseline_model(number_of_classes: int, dropout_rate: float = 0.40) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = create_augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_mobilenet_model(
    number_of_classes: int,
    dropout_rate: float,
    learning_rate: float,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = create_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenet_v2_transfer")
    compile_model(model, learning_rate)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def create_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7
        ),
    ]


def save_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    pd.DataFrame(history.history).to_csv(output_path, index_label="epoch")


def tune_mobilenet(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    number_of_classes: int,
    output_directory: Path,
    epochs: int,
) -> dict[str, float]:
    """Grid kecil: dua learning rate dan dua dropout."""
    candidates = [
        {"learning_rate": 1e-3, "dropout_rate": 0.30},
        {"learning_rate": 1e-3, "dropout_rate": 0.50},
        {"learning_rate": 1e-4, "dropout_rate": 0.30},
        {"learning_rate": 1e-4, "dropout_rate": 0.50},
    ]
    tuning_results = []

    for trial_number, candidate in enumerate(candidates, start=1):
        print(f"\nTuning {trial_number}/{len(candidates)}: {candidate}")
        tf.keras.backend.clear_session()
        model, _ = build_mobilenet_model(number_of_classes, **candidate)
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            verbose=1,
        )
        tuning_results.append(
            {
                **candidate,
                "best_validation_accuracy": float(max(history.history["val_accuracy"])),
                "best_validation_loss": float(min(history.history["val_loss"])),
            }
        )

    results_dataframe = pd.DataFrame(tuning_results).sort_values(
        ["best_validation_accuracy", "best_validation_loss"],
        ascending=[False, True],
    )
    results_dataframe.to_csv(output_directory / "tuning_results.csv", index=False)
    best = results_dataframe.iloc[0]
    return {
        "learning_rate": float(best["learning_rate"]),
        "dropout_rate": float(best["dropout_rate"]),
    }


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    model_name: str,
    output_directory: Path,
) -> dict[str, float]:
    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision_macro": float(
            precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "test_images": int(len(true_labels)),
    }
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    (output_directory / f"{model_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_directory / f"{model_name}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Kelas prediksi")
    plt.ylabel("Kelas sebenarnya")
    plt.title(f"Confusion matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_directory / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close()

    print(f"\nHasil test {model_name}: {metrics}")
    return metrics


def train_baseline(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    models_directory: Path,
    output_directory: Path,
    epochs: int,
) -> None:
    tf.keras.backend.clear_session()
    model = build_baseline_model(len(class_names))
    compile_model(model, learning_rate=1e-3)
    model_path = models_directory / "baseline_cnn.keras"
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=create_callbacks(model_path),
    )
    model = tf.keras.models.load_model(model_path)
    save_history(history, output_directory / "baseline_history.csv")
    evaluate_model(model, test_dataset, class_names, "baseline", output_directory)


def train_mobilenet(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    models_directory: Path,
    output_directory: Path,
    epochs: int,
    fine_tune_epochs: int,
) -> None:
    tuning_file = output_directory / "tuning_results.csv"
    if tuning_file.exists():
        best_row = pd.read_csv(tuning_file).sort_values(
            ["best_validation_accuracy", "best_validation_loss"],
            ascending=[False, True],
        ).iloc[0]
        parameters = {
            "learning_rate": float(best_row["learning_rate"]),
            "dropout_rate": float(best_row["dropout_rate"]),
        }
    else:
        parameters = {"learning_rate": 1e-3, "dropout_rate": 0.40}

    tf.keras.backend.clear_session()
    model, base_model = build_mobilenet_model(len(class_names), **parameters)
    model_path = models_directory / "mobilenet_v2_best.keras"
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=create_callbacks(model_path),
    )
    save_history(history, output_directory / "mobilenet_frozen_history.csv")

    # Fine-tuning hanya membuka 30 layer terakhir dengan learning rate kecil.
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    compile_model(model, learning_rate=parameters["learning_rate"] * 0.1)
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=fine_tune_epochs,
        callbacks=create_callbacks(model_path),
    )
    save_history(
        fine_tune_history, output_directory / "mobilenet_fine_tune_history.csv"
    )

    best_model = tf.keras.models.load_model(model_path)
    evaluate_model(best_model, test_dataset, class_names, "mobilenet", output_directory)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training CNN daun mangga.")
    parser.add_argument(
        "--model",
        choices=["prepare", "baseline", "tune", "mobilenet", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--tuning-epochs", type=int, default=4)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--zip", type=Path, default=Path("archive.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_random_seed()

    data_directory = Path("data/raw")
    models_directory = Path("models")
    output_directory = Path("outputs")
    models_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not args.zip.is_file() and not data_directory.exists():
        raise SystemExit(f"Dataset tidak ditemukan: {args.zip.resolve()}")
    if args.zip.is_file():
        safe_extract_zip(args.zip, data_directory)

    dataset_root = find_dataset_root(data_directory)
    split_dataframe = prepare_splits(dataset_root, output_directory)
    print_split_summary(split_dataframe)

    class_names = sorted(split_dataframe["class_name"].unique().tolist())
    (models_directory / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_dataset = create_dataset(split_dataframe, "train", class_names)
    validation_dataset = create_dataset(split_dataframe, "validation", class_names)
    test_dataset = create_dataset(split_dataframe, "test", class_names)

    if args.model in {"baseline", "all"}:
        train_baseline(
            train_dataset,
            validation_dataset,
            test_dataset,
            class_names,
            models_directory,
            output_directory,
            args.epochs,
        )

    if args.model in {"tune", "all"}:
        best_parameters = tune_mobilenet(
            train_dataset,
            validation_dataset,
            len(class_names),
            output_directory,
            args.tuning_epochs,
        )
        print(f"Parameter terbaik: {best_parameters}")

    if args.model in {"mobilenet", "all"}:
        train_mobilenet(
            train_dataset,
            validation_dataset,
            test_dataset,
            class_names,
            models_directory,
            output_directory,
            args.epochs,
            args.fine_tune_epochs,
        )


if __name__ == "__main__":
    main()
