import os
import time
import traceback

import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify

import tensorflow as tf


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Allow uploads up to 10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CUSTOM TRANSFORMER LAYERS
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
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_skin_lesion_model.keras"
)


model = None
model_load_error = None


print("\n========================================", flush=True)
print("DERMAI STARTING", flush=True)
print("========================================", flush=True)

print(
    f"MODEL PATH: {MODEL_PATH}",
    flush=True
)

print(
    f"MODEL EXISTS: {os.path.exists(MODEL_PATH)}",
    flush=True
)


try:

    print(
        "LOADING MODEL...",
        flush=True
    )


    model = tf.keras.models.load_model(

        MODEL_PATH,

        compile=False,

        custom_objects={

            "Patches": Patches,

            "PatchEncoder": PatchEncoder

        }

    )


    print("\n========================================", flush=True)

    print(
        "MODEL LOADED SUCCESSFULLY",
        flush=True
    )

    print(
        f"MODEL INPUT: {model.input_shape}",
        flush=True
    )

    print(
        f"MODEL OUTPUT: {model.output_shape}",
        flush=True
    )

    print("========================================\n", flush=True)


except Exception as e:

    model_load_error = traceback.format_exc()

    print("\n========================================", flush=True)

    print(
        "MODEL LOAD ERROR",
        flush=True
    )

    print(
        model_load_error,
        flush=True
    )

    print("========================================\n", flush=True)



# ============================================================
# WARM UP MODEL
# ============================================================

if model is not None:

    try:

        print(
            "WARMING UP MODEL...",
            flush=True
        )


        dummy_input = np.zeros(
            (1, 224, 224, 3),
            dtype=np.float32
        )


        # Run one prediction during startup
        # This prevents first real request issues

        _ = model(
            dummy_input,
            training=False
        )


        print(
            "MODEL WARM-UP SUCCESSFUL",
            flush=True
        )


    except Exception:

        print(
            "MODEL WARM-UP FAILED:",
            flush=True
        )

        print(
            traceback.format_exc(),
            flush=True
        )



# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_img):

    print(
        "PREPROCESSING IMAGE...",
        flush=True
    )


    # Convert image to RGB

    image = pil_img.convert("RGB")


    # Resize

    image = image.resize(
        (224, 224)
    )


    # Convert to NumPy

    image_array = np.array(
        image,
        dtype=np.float32
    )


    print(
        f"RAW IMAGE SHAPE: {image_array.shape}",
        flush=True
    )


    # IMPORTANT:
    # Normalize image between 0 and 1
    # If your training code used rescaling

    image_array = image_array / 255.0


    # Add batch dimension

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    print(
        f"FINAL INPUT SHAPE: {image_array.shape}",
        flush=True
    )


    print(
        f"INPUT MIN: {image_array.min()}",
        flush=True
    )


    print(
        f"INPUT MAX: {image_array.max()}",
        flush=True
    )


    return image_array



# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )



# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    print("\n========================================", flush=True)

    print(
        "ANALYZE REQUEST RECEIVED",
        flush=True
    )

    print("========================================", flush=True)


    try:

        # ====================================================
        # CHECK MODEL
        # ====================================================

        if model is None:

            print(
                "ERROR: MODEL NOT LOADED",
                flush=True
            )


            return jsonify({

                "error":
                    "AI model is unavailable.",

                "details":
                    model_load_error

            }), 500


        # ====================================================
        # CHECK IMAGE
        # ====================================================

        if "image" not in request.files:

            print(
                "ERROR: NO IMAGE IN REQUEST",
                flush=True
            )


            return jsonify({

                "error":
                    "No image was uploaded."

            }), 400


        file = request.files["image"]


        print(
            f"FILENAME: {file.filename}",
            flush=True
        )


        if not file.filename:

            return jsonify({

                "error":
                    "No image selected."

            }), 400


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        print(
            "OPENING IMAGE...",
            flush=True
        )


        image = Image.open(
            file.stream
        )


        print(
            f"IMAGE SIZE: {image.size}",
            flush=True
        )


        print(
            f"IMAGE MODE: {image.mode}",
            flush=True
        )


        # ====================================================
        # PREPROCESS
        # ====================================================

        processed_image = preprocess_image(
            image
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        print(
            "STARTING AI PREDICTION...",
            flush=True
        )


        start_time = time.time()


        # DO NOT USE model.predict()
        # Direct model call is lighter and more stable on Render

        prediction_tensor = model(
            processed_image,
            training=False
        )


        prediction = prediction_tensor.numpy()


        elapsed = time.time() - start_time


        print(
            f"PREDICTION FINISHED IN {elapsed:.2f} SECONDS",
            flush=True
        )


        print(
            f"RAW PREDICTION: {prediction}",
            flush=True
        )


        # ====================================================
        # GET PROBABILITY
        # ====================================================

        probability = float(
            np.asarray(
                prediction
            ).flatten()[0]
        )


        print(
            f"PROBABILITY: {probability}",
            flush=True
        )


        # ====================================================
        # SAFETY CLAMP
        # ====================================================

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )


        # ====================================================
        # INTERPRET RESULT
        # ====================================================

        if probability >= 0.5:

            label = "Malignant"

            confidence_value = probability

        else:

            label = "Benign"

            confidence_value = 1 - probability


        confidence_percent = round(
            confidence_value * 100,
            2
        )


        print(
            f"RESULT: {label}",
            flush=True
        )


        print(
            f"CONFIDENCE: {confidence_percent}%",
            flush=True
        )


        # ====================================================
        # RETURN JSON
        # ====================================================

        response_data = {

            "prediction": label,

            "confidence": confidence_percent,

            "inference_time":
                round(
                    elapsed,
                    2
                )

        }


        print(
            f"RETURNING: {response_data}",
            flush=True
        )


        print("========================================\n", flush=True)


        return jsonify(
            response_data
        )


    except Exception as e:

        error_details = traceback.format_exc()


        print("\n========================================", flush=True)

        print(
            "ANALYSIS ERROR",
            flush=True
        )

        print(
            error_details,
            flush=True
        )

        print("========================================\n", flush=True)


        return jsonify({

            "error":
                "Analysis failed.",

            "details":
                str(e)

        }), 500



# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if model is None:

        return jsonify({

            "status": "error",

            "model_loaded": False,

            "error": model_load_error

        }), 500


    return jsonify({

        "status": "ok",

        "model_loaded": True,

        "model_input":
            str(model.input_shape),

        "model_output":
            str(model.output_shape)

    })



# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "error":
            "Image file is too large. Maximum size is 10MB."

    }), 413



# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print(
        f"STARTING SERVER ON PORT {port}",
        flush=True
    )


    app.run(

        host="0.0.0.0",

        port=port

    )
