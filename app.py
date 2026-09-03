# ============================================================
# DERMAI - FLASK APPLICATION
# Optimized for low-memory deployment
# ============================================================

import os

# ------------------------------------------------------------
# IMPORTANT: Set TensorFlow environment variables BEFORE
# importing TensorFlow
# ------------------------------------------------------------

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import time
import gc
import traceback

import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify

import tensorflow as tf


# ============================================================
# LIMIT TENSORFLOW RESOURCE USAGE
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CUSTOM VISION TRANSFORMER LAYERS
# ============================================================

class Patches(tf.keras.layers.Layer):

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):

        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        patch_dims = patches.shape[-1]

        patches = tf.reshape(
            patches,
            [batch_size, -1, patch_dims]
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
            projection_dim
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
            + self.position_embedding(positions)
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


print("=" * 60, flush=True)
print("DERMAI STARTING", flush=True)
print("=" * 60, flush=True)

print(f"MODEL PATH: {MODEL_PATH}", flush=True)
print(f"MODEL EXISTS: {os.path.exists(MODEL_PATH)}", flush=True)


try:

    print("LOADING AI MODEL...", flush=True)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder
        }
    )

    print("MODEL LOADED SUCCESSFULLY!", flush=True)
    print(f"MODEL INPUT: {model.input_shape}", flush=True)
    print(f"MODEL OUTPUT: {model.output_shape}", flush=True)

except Exception:

    model_load_error = traceback.format_exc()

    print("MODEL LOAD ERROR:", flush=True)
    print(model_load_error, flush=True)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_img):

    print("PREPROCESSING IMAGE...", flush=True)

    # Convert to RGB
    image = pil_img.convert("RGB")

    # Resize exactly as training input expects
    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    # Convert to float32
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array /= 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make memory layout efficient
    image_array = np.ascontiguousarray(
        image_array,
        dtype=np.float32
    )

    print(
        f"FINAL INPUT SHAPE: {image_array.shape}",
        flush=True
    )

    return image_array


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    print("\n" + "=" * 60, flush=True)
    print("ANALYZE REQUEST RECEIVED", flush=True)
    print("=" * 60, flush=True)

    processed_image = None
    prediction_tensor = None

    try:

        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "error": "AI model failed to load."
            }), 500


        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded."
            }), 400


        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "No image selected."
            }), 400


        print(
            f"PROCESSING FILE: {file.filename}",
            flush=True
        )


        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(file.stream)

        print(
            f"ORIGINAL SIZE: {image.size}",
            flush=True
        )


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_image = preprocess_image(image)


        # Explicitly close image
        image.close()


        # ----------------------------------------------------
        # MEMORY CLEANUP BEFORE INFERENCE
        # ----------------------------------------------------

        gc.collect()


        # ----------------------------------------------------
        # RUN INFERENCE
        # ----------------------------------------------------

        print(
            "STARTING AI PREDICTION...",
            flush=True
        )

        start_time = time.time()


        # Use direct TensorFlow inference
        prediction_tensor = model(
            processed_image,
            training=False
        )


        # Convert immediately to NumPy
        prediction = prediction_tensor.numpy()


        elapsed = time.time() - start_time


        print(
            f"PREDICTION FINISHED IN {elapsed:.2f}s",
            flush=True
        )


        print(
            f"RAW OUTPUT: {prediction}",
            flush=True
        )


        # ----------------------------------------------------
        # EXTRACT PROBABILITY
        # ----------------------------------------------------

        probability = float(
            np.asarray(prediction)
            .flatten()[0]
        )


        # Safety clamp
        probability = max(
            0.0,
            min(1.0, probability)
        )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if probability >= 0.5:

            label = "Malignant"
            confidence = probability

        else:

            label = "Benign"
            confidence = 1.0 - probability


        confidence_percent = round(
            confidence * 100,
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


        # ----------------------------------------------------
        # CLEAN RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "prediction": label,

            "confidence": confidence_percent,

            "inference_time": round(
                elapsed,
                2
            )

        })


    except Exception as e:

        error_details = traceback.format_exc()

        print("\nANALYSIS ERROR:", flush=True)
        print(error_details, flush=True)

        return jsonify({

            "error": "Analysis failed.",

            "details": str(e)

        }), 500


    finally:

        # ----------------------------------------------------
        # FREE TEMPORARY MEMORY
        # ----------------------------------------------------

        try:
            del processed_image
        except Exception:
            pass

        try:
            del prediction_tensor
        except Exception:
            pass

        gc.collect()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy" if model is not None else "error",

        "model_loaded":
            model is not None

    })


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "error":
            "Image too large. Maximum file size is 10MB."

    }), 413


# ============================================================
# START LOCAL SERVER
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
        port=port,
        debug=False
    )
