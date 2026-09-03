import os
import io
import base64
import numpy as np

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template, jsonify

import tensorflow as tf
from tensorflow.keras import layers


# =========================================================
# CUSTOM LAYER 1: PATCHES
# Required by your Transformer model
# =========================================================

class Patches(layers.Layer):

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):

        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            strides=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        patch_dims = patches.shape[-1]

        patches = tf.reshape(
            patches,
            [
                batch_size,
                -1,
                patch_dims
            ]
        )

        return patches

    def get_config(self):

        config = super().get_config()

        config.update({
            "patch_size": self.patch_size
        })

        return config


# =========================================================
# CUSTOM LAYER 2: PATCH ENCODER
# Required by your Transformer model
# =========================================================

class PatchEncoder(layers.Layer):

    def __init__(
        self,
        num_patches,
        projection_dim,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = layers.Dense(
            projection_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim
        )


    def call(self, patch):

        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )

        encoded = (
            self.projection(patch)
            +
            self.position_embedding(positions)
        )

        return encoded


    def get_config(self):

        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })

        return config


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_skin_lesion_model.keras"
)


# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = (
    224,
    224
)


# =========================================================
# LOAD YOUR TRAINED AI MODEL
# =========================================================

print("Loading AI model...")


model = tf.keras.models.load_model(

    MODEL_PATH,

    custom_objects={

        "Patches": Patches,

        "PatchEncoder": PatchEncoder

    },

    compile=False

)


print("AI model loaded successfully!")


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(file_stream):

    # Open uploaded image
    image = Image.open(
        file_stream
    ).convert("RGB")


    # Keep original for website display
    original = image.copy()


    # Resize for AI model
    image = image.resize(
        IMG_SIZE
    )


    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )


    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array, original


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# IMAGE ANALYSIS
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)

def analyze():


    # Check image exists
    if "image" not in request.files:

        return jsonify({

            "error":
            "No image was uploaded."

        }), 400


    file = request.files["image"]


    # Check filename
    if file.filename == "":

        return jsonify({

            "error":
            "Please select an image."

        }), 400


    try:

        processed_image, original_image = preprocess_image(
            file.stream
        )


    except (
        UnidentifiedImageError,
        OSError
    ):

        return jsonify({

            "error":
            "Invalid image file."

        }), 400


    # =====================================================
    # AI PREDICTION
    # =====================================================

    prediction = model.predict(

        processed_image,

        verbose=0

    )


    # Convert prediction to single number
    prediction = float(

        prediction.flatten()[0]

    )


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if prediction >= 0.5:

        label = "Malignant"

        confidence = prediction

        risk = "high"


    else:

        label = "Benign"

        confidence = 1 - prediction

        risk = "low"


    # =====================================================
    # CONVERT IMAGE TO BASE64
    # So website can display it
    # =====================================================

    buffer = io.BytesIO()


    original_image.save(

        buffer,

        format="JPEG",

        quality=90

    )


    image_base64 = base64.b64encode(

        buffer.getvalue()

    ).decode("utf-8")


    # =====================================================
    # SEND RESULTS TO WEBSITE
    # =====================================================

    return jsonify({

        "label": label,

        "confidence": round(
            confidence * 100,
            2
        ),

        "raw_prediction": round(
            prediction,
            5
        ),

        "risk": risk,

        "image": image_base64

    })


# =========================================================
# RUN WEBSITE
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )