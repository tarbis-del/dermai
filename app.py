import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf


app = Flask(__name__)


# ============================================================
# 1. CUSTOM LAYERS
# ============================================================

class Patches(tf.keras.layers.Layer):

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
            rates=[
                1,
                1,
                1,
                1
            ],
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


class PatchEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        num_patches,
        projection_dim,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = tf.keras.layers.Dense(
            units=projection_dim
        )

        self.position_embedding = tf.keras.layers.Embedding(
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


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

MODEL_PATH = "best_skin_lesion_model.keras"

model = None
model_load_error = None


try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder
        }
    )

    print("========================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("========================================")


except Exception as e:

    model_load_error = str(e)

    print("========================================")
    print("MODEL LOAD ERROR")
    print(repr(e))
    print("========================================")


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_img):

    # Convert image to RGB
    img = pil_img.convert("RGB")

    # Resize to the model's expected size
    img = img.resize(
        (224, 224)
    )

    # Convert to NumPy array
    arr = np.array(
        img,
        dtype=np.float32
    )

    # Add batch dimension
    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


# ============================================================
# 4. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# 5. IMAGE ANALYSIS
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # Check that the model loaded
    if model is None:

        return jsonify({
            "error":
                "AI model could not be loaded.",

            "details":
                model_load_error
        }), 500


    # Check that an image was uploaded
    if "image" not in request.files:

        return jsonify({
            "error":
                "No image was uploaded."
        }), 400


    file = request.files["image"]


    # Check that a file was actually selected
    if file.filename == "":

        return jsonify({
            "error":
                "No image was selected."
        }), 400


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    try:

        image = Image.open(
            file.stream
        )

        processed_image = preprocess_image(
            image
        )

    except Exception as e:

        return jsonify({

            "error":
                "The uploaded file could not be processed.",

            "details":
                str(e)

        }), 400


    # ========================================================
    # RUN AI MODEL
    # ========================================================

    try:

        prediction = model.predict(
            processed_image,
            verbose=0
        )

        probability = float(
            np.asarray(
                prediction
            ).flatten()[0]
        )


        # ====================================================
        # INTERPRET MODEL OUTPUT
        # ====================================================

        if probability > 0.5:

            label = "Malignant"

            confidence = probability

        else:

            label = "Benign"

            confidence = 1 - probability


        # ====================================================
        # RETURN RESULT TO WEBSITE
        # ====================================================

        return jsonify({

            "prediction":
                label,

            "confidence":
                round(
                    confidence * 100,
                    2
                )

        })


    # ========================================================
    # CATCH MODEL ERRORS
    # ========================================================

    except Exception as e:

        print(
            "MODEL PREDICTION ERROR:",
            repr(e)
        )

        return jsonify({

            "error":
                "The AI model could not analyze this image.",

            "details":
                str(e)

        }), 500


# ============================================================
# 6. START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
