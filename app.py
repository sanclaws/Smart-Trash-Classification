from flask import Flask, render_template, request, redirect
import tensorflow as tf
import numpy as np
import os
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trash_model.keras")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class_names = np.load("classes.npy", allow_pickle=True)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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
        )
    }

    return advice_map.get(
        label.strip().lower(),
        "Pisahkan sampah sesuai jenisnya agar lebih mudah didaur ulang."
    )


def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    label = class_names[index]

    advice = get_advice(label)

    return label, confidence, advice


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file and file.filename != "":
        filename = str(uuid.uuid4()) + ".jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        result, confidence, advice = predict_image(save_path)
        img_path = "uploads/" + filename

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            img_path=img_path,
            advice=advice
        )

    return redirect("/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8585))
    app.run(host="0.0.0.0", port=port, debug=True)