import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("trash_model.keras")

class_names = np.load("classes.npy", allow_pickle=True)

img = tf.keras.utils.load_img("kaleng.jpg", target_size=(224, 224))
img = tf.keras.utils.img_to_array(img)
img = np.expand_dims(img, axis=0)

img = tf.keras.applications.efficientnet.preprocess_input(img)

pred = model.predict(img)

index = np.argmax(pred)
confidence = np.max(pred) * 100

print("Hasil:", class_names[index])
print("Confidence:", confidence, "%")
print("RAW:", pred)
