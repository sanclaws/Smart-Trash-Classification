from flask import Flask, render_template, request, redirect
from PIL import Image, ImageOps, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import os
import io
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trash_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

# Vercel serves files in /public directly. Flask's own static route is disabled.
app = Flask(__name__, static_folder=None, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB per uploaded image

# Load the trained model and class labels once per serverless function instance.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = np.load(CLASSES_PATH, allow_pickle=True)


def get_advice(label):
    advice_map = {
        "organik": (
            "Sampah organik dapat terurai secara alami. "
            "Buang ke tempat sampah organik atau olah menjadi kompos "
            "agar dapat dimanfaatkan sebagai pupuk!"
        ),
        "plastik": (
            "Sampah plastik sebaiknya dibersihkan terlebih dahulu "
            "agar dapat didaur ulang. Pisahkan ke tempat sampah "
            "anorganik atau bank sampah plastik!"
        ),
        "kertas": (
            "Sampah kertas dapat didaur ulang menjadi produk baru. "
            "Pastikan kertas dalam kondisi kering dan bersih sebelum dibuang "
            "ke tempat sampah kertas!"
        ),
        "kaca": (
            "Sampah kaca harus dibuang dengan hati-hati karena dapat melukai. "
            "Pisahkan ke tempat khusus kaca atau daur ulang agar aman diproses!"
        ),
        "logam": (
            "Sampah besi atau logam dapat didaur ulang dan memiliki nilai jual. "
            "Kumpulkan lalu serahkan ke tempat daur ulang logam atau bank sampah!"
        ),
    }
    return advice_map.get(
        str(label).strip().lower(),
        "Pisahkan sampah sesuai jenisnya agar lebih mudah didaur ulang.",
    )


def predict_image(image):
    """Run the existing Keras model on a PIL image."""
    resized = image.resize((224, 224))
    image_array = tf.keras.utils.img_to_array(resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)

    prediction = model.predict(image_array, verbose=0)
    index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100
    label = str(class_names[index])
    return label, confidence, get_advice(label)


def make_preview_data_uri(image):
    """Create a small JPEG preview in memory; no local upload storage needed."""
    preview = image.copy()
    preview.thumbnail((1000, 1000))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return redirect("/")

    try:
        image = Image.open(uploaded_file.stream)
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template(
            "index.html",
            error="File yang dipilih bukan gambar yang valid. Silakan pilih gambar JPG atau PNG.",
        ), 400

    result, confidence, advice = predict_image(image)
    img_data = make_preview_data_uri(image)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        img_data=img_data,
        advice=advice,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="Ukuran gambar terlalu besar. Maksimal ukuran file adalah 4 MB.",
    ), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
