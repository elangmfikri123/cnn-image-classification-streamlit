"""Aplikasi Streamlit untuk klasifikasi penyakit daun mangga."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


MODEL_PATH = Path("models/mobilenet_v2_best.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="Klasifikasi penyakit daun mangga",
    page_icon=":material/eco:",
    layout="centered",
)


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def load_model(model_path: str) -> tf.keras.Model:
    """Memuat model satu kali lalu menggunakannya kembali pada setiap rerun."""
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names(class_names_path: str) -> list[str]:
    return json.loads(Path(class_names_path).read_text(encoding="utf-8"))


def prepare_image(image: Image.Image) -> np.ndarray:
    """Menyamakan orientasi, warna, dan ukuran dengan input saat training."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def predict_image(
    model: tf.keras.Model, image: Image.Image, class_names: list[str]
) -> pd.DataFrame:
    input_batch = prepare_image(image)
    probabilities = model.predict(input_batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    return pd.DataFrame(
        {
            "Kelas": [class_names[index] for index in top_indices],
            "Probabilitas": [float(probabilities[index]) for index in top_indices],
        }
    )


st.title("Klasifikasi penyakit daun mangga")
st.write(
    "Unggah foto daun mangga untuk memperoleh kelas prediksi dan tiga "
    "probabilitas tertinggi dari model MobileNetV2."
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Pilih foto daun",
        type=["jpg", "jpeg", "png"],
        max_upload_size=10,
        help="Format yang diterima: JPG, JPEG, atau PNG. Maksimum 10 MB.",
        key="leaf_image",
    )

if uploaded_file is None:
    st.info("Unggah sebuah foto untuk memulai prediksi.", icon=":material/upload_file:")
    st.stop()

try:
    uploaded_image = Image.open(uploaded_file)
    uploaded_image.load()
except (UnidentifiedImageError, OSError, ValueError):
    st.error("File tidak dapat dibaca sebagai gambar yang valid.")
    st.stop()

with st.container(border=True):
    st.subheader("Gambar yang diunggah")
    st.image(uploaded_image, caption=uploaded_file.name, width="stretch")

if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
    st.error(
        "Model belum tersedia. Jalankan `python train.py --model mobilenet` "
        "terlebih dahulu."
    )
    st.stop()

with st.spinner("Menganalisis gambar daun...", show_time=True):
    try:
        classification_model = load_model(str(MODEL_PATH))
        class_names = load_class_names(str(CLASS_NAMES_PATH))
        top_predictions = predict_image(
            classification_model, uploaded_image, class_names
        )
    except Exception as error:
        st.error(f"Prediksi gagal dilakukan: {error}")
        st.stop()

best_prediction = top_predictions.iloc[0]
with st.container(border=True):
    st.subheader("Hasil prediksi")
    st.metric(
        "Kelas utama",
        best_prediction["Kelas"],
        help="Kelas dengan probabilitas tertinggi menurut model.",
    )
    st.metric(
        "Tingkat keyakinan",
        f"{best_prediction['Probabilitas'] * 100:.2f}%",
    )

    st.write("**Tiga kelas dengan probabilitas tertinggi**")
    display_predictions = top_predictions.copy()
    display_predictions["Probabilitas"] = (
        display_predictions["Probabilitas"] * 100
    ).round(2)
    st.dataframe(
        display_predictions,
        hide_index=True,
        key="prediction_probabilities",
        column_config={
            "Probabilitas": st.column_config.ProgressColumn(
                "Probabilitas (%)", min_value=0.0, max_value=100.0, format="%.2f%%"
            )
        },
    )

st.caption(
    "Prediksi ini merupakan hasil model pembelajaran mesin dan bukan diagnosis "
    "resmi. Gunakan pemeriksaan ahli untuk keputusan penanganan tanaman."
)
