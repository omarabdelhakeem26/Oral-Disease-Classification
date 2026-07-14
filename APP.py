import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# =========================
# Load Model
# =========================
print("Loading EfficientNetB0 model...")
base_model = EfficientNetB0(
    weights=None,
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False
print("EfficientNetB0 model loaded successfully.")
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(6, activation="softmax")
])

print("Loading model weights...")
try:
    model.load_weights("efficientnet.weights.h5")
    print("Model weights loaded successfully.")
except Exception as e:
    print("LOAD ERROR:")
    print(type(e).__name__)
    print(e)
print("Model weights loaded successfully.")
# =========================
# Classes
# =========================
class_names = [
    "Calculus",
    "Dental Caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "Hypodontia"
]

# =========================
# Disease Description
# =========================
disease_info = {
    "Calculus": "Dental calculus (tartar) is hardened dental plaque that forms on teeth.",
    "Dental Caries": "Dental caries (tooth decay) is caused by bacteria damaging tooth enamel.",
    "Gingivitis": "Gingivitis is inflammation of the gums caused by plaque buildup.",
    "Mouth Ulcer": "Mouth ulcers are painful sores inside the mouth.",
    "Tooth Discoloration": "Tooth discoloration refers to changes in the natural color of teeth.",
    "Hypodontia": "Hypodontia is a condition where one or more teeth fail to develop."
}

# =========================
# Prediction Function
# =========================
def predict(image):

    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]

    predicted_class = np.argmax(prediction)

    disease = class_names[predicted_class]

    confidence = float(prediction[predicted_class] * 100)

    probs = {
        class_names[i]: float(prediction[i])
        for i in range(len(class_names))
    }

    return (
        disease,
        f"{confidence:.2f} %",
        disease_info[disease],
        probs
    )
# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
# 🦷 Oral Disease Classification

### Deep Learning using EfficientNetB0

Upload an oral image to classify the disease.
"""
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Oral Image"
            )

            predict_btn = gr.Button(
                "Predict",
                variant="primary"
            )

        with gr.Column():

            disease = gr.Textbox(
                label="Predicted Disease"
            )

            confidence = gr.Textbox(
                label="Confidence"
            )

            description = gr.Textbox(
                label="Disease Description",
                lines=4
            )

            probabilities = gr.Label(
                label="Prediction Probabilities",
                num_top_classes=3
            )

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            disease,
            confidence,
            description,
            probabilities
        ]
    )
print("Launching Gradio app...")
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True,
    debug=True
)
print("Gradio app launched successfully.")