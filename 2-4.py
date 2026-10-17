
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI
# ==========================================

OUTPUT_DIR = Path("outputs")
MODEL_PATH = Path("models/mobilenet_v2_best.keras")
CLASS_PATH = Path("models/class_names.json")

SPLIT_PATH = OUTPUT_DIR / "data_splits.csv"
METRICS_PATH = OUTPUT_DIR / "mobilenet_metrics.json"
REPORT_PATH = OUTPUT_DIR / "mobilenet_classification_report.json"
CM_PATH = OUTPUT_DIR / "mobilenet_confusion_matrix.png"

HISTORY_FILES = [
    OUTPUT_DIR / "mobilenet_frozen_history.csv",
    OUTPUT_DIR / "mobilenet_fine_tune_history.csv",
]

# ==========================================
# FUNGSI PEMBANTU
# ==========================================

def load_json(path):
    if not path.exists():
        print(f"[INFO] File belum tersedia: {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[PERINGATAN] Gagal membaca {path}: {error}")
        return None


def find_column(df, candidates):
    """Mencari nama kolom dengan beberapa kemungkinan nama."""
    columns = {str(col).lower().strip(): col for col in df.columns}

    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]

    return None


def print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ==========================================
# 1. HASIL PREPROCESSING DAN PEMBAGIAN DATA
# ==========================================

def show_preprocessing():
    print("\n" + "=" * 65)
    print("1. HASIL PREPROCESSING DAN PEMBAGIAN DATA")
    print("=" * 65)

    print("Ukuran input model: 224 x 224 piksel")
    print("Kanal gambar: RGB")

    if CLASS_PATH.exists():
        class_names = load_json(CLASS_PATH)
        if isinstance(class_names, list):
            print(f"Jumlah kelas: {len(class_names)}")
            print("Kelas:", ", ".join(class_names))

    if not SPLIT_PATH.exists():
        print(f"\nFile pembagian data tidak ditemukan: {SPLIT_PATH}")
        print("Pastikan tahap prepare_splits sudah dijalankan.")
        return

    df = pd.read_csv(SPLIT_PATH)

    print(f"\nJumlah baris pada data_splits.csv: {len(df)}")
    print("Kolom:", list(df.columns))

    # Cari kolom split dan kelas secara otomatis.
    split_col = find_column(
        df, ["split", "subset", "set", "data_split", "partition"]
    )
    class_col = find_column(
        df, ["class_name", "class", "label", "category", "kelas"]
    )

    # Jika kelas tidak tersedia sebagai kolom, coba ambil dari path.
    path_col = find_column(
        df, ["path", "filepath", "file_path", "image_path", "filename"]
    )

    if class_col is None and path_col is not None:
        df["_detected_class"] = df[path_col].astype(str).apply(
            lambda p: Path(p).parent.name
        )
        class_col = "_detected_class"

    if split_col is not None:
        print("\nJumlah data per subset:")
        print(df[split_col].value_counts(dropna=False).to_string())

        print("\nPersentase setiap subset:")
        percentages = (
            df[split_col].value_counts(normalize=True, dropna=False) * 100
        ).round(2)
        print(percentages.to_string())

    if class_col is not None:
        print("\nJumlah data per kelas:")
        if split_col is not None:
            table = pd.crosstab(
                df[class_col], df[split_col], margins=True
            )
            print(table.to_string())
        else:
            print(df[class_col].value_counts().to_string())

    if split_col is not None and class_col is not None:
        counts = pd.crosstab(df[class_col], df[split_col])
        counts.plot(kind="bar", figsize=(12, 6))
        plt.title("Distribusi Dataset per Kelas dan Subset")
        plt.xlabel("Kelas")
        plt.ylabel("Jumlah gambar")
        plt.xticks(rotation=40, ha="right")
        plt.legend(title="Subset")
        plt.tight_layout()

        chart_path = OUTPUT_DIR / "preprocessing_distribution.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"\nGrafik preprocessing disimpan: {chart_path}")
        plt.show()
        plt.close()


# ==========================================
# 2. HASIL TRAINING
# ==========================================

def show_training():
    print("\n" + "=" * 65)
    print("2. HASIL TRAINING MODEL MOBILENETV2")
    print("=" * 65)

    histories = []

    for path in HISTORY_FILES:
        if not path.exists():
            print(f"[INFO] Riwayat belum ditemukan: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            print(f"[PERINGATAN] Gagal membaca {path}: {error}")
            continue

        print(f"\nFile: {path.name}")
        print(f"Jumlah epoch tercatat: {len(df)}")
        print("Kolom:", list(df.columns))

        # Hapus kolom indeks CSV yang tidak diperlukan.
        for col in list(df.columns):
            if col.lower().startswith("unnamed"):
                df = df.drop(columns=col)

        df["_stage"] = path.stem
        histories.append(df)

        for metric in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            if metric in df.columns and not df[metric].dropna().empty:
                values = pd.to_numeric(df[metric], errors="coerce")
                valid = values.dropna()

                if not valid.empty:
                    print(
                        f"  {metric}: awal={valid.iloc[0]:.4f}, "
                        f"akhir={valid.iloc[-1]:.4f}, "
                        f"terbaik={valid.max() if 'accuracy' in metric else valid.min():.4f}"
                    )

    if not histories:
        print(
            "\nRiwayat training belum tersedia. "
            "Pastikan train.py menyimpan file history CSV."
        )
        return

    history = pd.concat(histories, ignore_index=True)

    # Grafik accuracy.
    plt.figure(figsize=(10, 5))
    for stage, df in history.groupby("_stage", sort=False):
        for metric in ["accuracy", "val_accuracy"]:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors="coerce")
                label = f"{stage} - {metric}"
                plt.plot(
                    np.arange(1, len(values) + 1),
                    values,
                    marker="o",
                    label=label,
                )

    plt.title("Grafik Accuracy Training MobileNetV2")
    plt.xlabel("Epoch pada setiap tahap")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_path = OUTPUT_DIR / "mobilenet_training_accuracy.png"
    plt.savefig(accuracy_path, dpi=150, bbox_inches="tight")
    print(f"\nGrafik accuracy disimpan: {accuracy_path}")
    plt.show()
    plt.close()

    # Grafik loss.
    plt.figure(figsize=(10, 5))
    for stage, df in history.groupby("_stage", sort=False):
        for metric in ["loss", "val_loss"]:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors="coerce")
                plt.plot(
                    np.arange(1, len(values) + 1),
                    values,
                    marker="o",
                    label=f"{stage} - {metric}",
                )

    plt.title("Grafik Loss Training MobileNetV2")
    plt.xlabel("Epoch pada setiap tahap")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = OUTPUT_DIR / "mobilenet_training_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"Grafik loss disimpan: {loss_path}")
    plt.show()
    plt.close()


# ==========================================
# 3. HASIL PENGUJIAN
# ==========================================

def show_evaluation():
    print("\n" + "=" * 65)
    print("3. HASIL PENGUJIAN MODEL")
    print("=" * 65)

    if MODEL_PATH.exists():
        print(f"Model ditemukan: {MODEL_PATH}")
    else:
        print(f"[PERINGATAN] Model tidak ditemukan: {MODEL_PATH}")

    metrics = load_json(METRICS_PATH)

    if metrics is not None:
        print("\nMetrik evaluasi tersimpan:")
        print_json(metrics)

        # Tampilkan nilai umum jika nama key cocok.
        aliases = {
            "Accuracy": ["accuracy", "test_accuracy"],
            "Macro Precision": ["macro_precision", "precision_macro"],
            "Macro Recall": ["macro_recall", "recall_macro"],
            "Macro F1-score": ["macro_f1", "macro_f1_score", "f1_macro"],
            "Jumlah gambar testing": ["test_images", "test_samples"],
        }

        print("\nRingkasan metrik:")
        for label, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)) and label != "Jumlah gambar testing":
                        print(f"{label}: {value:.4f} ({value * 100:.2f}%)")
                    else:
                        print(f"{label}: {value}")
                    break
    else:
        print("File metrik utama belum tersedia.")

    report = load_json(REPORT_PATH)
    if report is not None:
        print("\nClassification report:")
        print_json(report)

    if CM_PATH.exists():
        image = plt.imread(CM_PATH)
        plt.figure(figsize=(8, 7))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Confusion Matrix MobileNetV2")
        plt.tight_layout()
        plt.show()
        plt.close()
        print(f"File confusion matrix: {CM_PATH}")
    else:
        print(f"[INFO] Confusion matrix tidak ditemukan: {CM_PATH}")


# ==========================================
# JALANKAN SEMUA PEMERIKSAAN
# ==========================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    show_preprocessing()
    show_training()
    show_evaluation()

    print("\n" + "=" * 65)
    print("PEMERIKSAAN SELESAI")
    print("=" * 65)
    print("Grafik yang berhasil dibuat tersimpan di folder outputs/.")


if __name__ == "__main__":
    main()
