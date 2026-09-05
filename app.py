
# ============================================================
# DERMAI - FLASK APPLICATION
# Optimized for low-memory deployment
# ============================================================

# IMPORTANT: Set these BEFORE importing TensorFlow
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import gc
import time
import traceback

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

import tensorflow as tf


# ============================================================
# LIMIT TENSORFLOW MEMORY / THREAD USAGE
# ============================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
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
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        patch_dims = tf.shape(patches)[-1]

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

        return (
            self.projection(patch)
            + self.position_embedding(positions)
        )

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_transformer_model.keras"
)

model = None
model_load_error = None

print("=" * 60, flush=True)
print("DERMAI STARTING - LOW MEMORY MODE", flush=True)
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

    print("MODEL LOADED SUCCESSFULLY", flush=True)
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

    image = pil_img.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# DIAGNOSTIC ROUTE
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_load_error
    })


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    print("ANALYZE REQUEST RECEIVED", flush=True)

    processed_image = None

    try:

        if model is None:

            return jsonify({
                "error": "AI model failed to load."
            }), 500


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
            f"PROCESSING: {file.filename}",
            flush=True
        )


        # Open image
        image = Image.open(file.stream)

        print(
            f"IMAGE SIZE: {image.size}",
            flush=True
        )


        # Preprocess
        processed_image = preprocess_image(image)

        print(
            f"INPUT SHAPE: {processed_image.shape}",
            flush=True
        )


        # Force garbage collection BEFORE prediction
        gc.collect()


        print(
            "STARTING PREDICTION...",
            flush=True
        )


        start_time = time.time()


        # Direct inference
        prediction_tensor = model(
            processed_image,
            training=False
        )


        prediction = prediction_tensor.numpy()


        elapsed = time.time() - start_time


        print(
            f"PREDICTION COMPLETE: {elapsed:.2f}s",
            flush=True
        )


        print(
            f"RAW OUTPUT: {prediction}",
            flush=True
        )


        probability = float(
            prediction.flatten()[0]
        )


        # Clamp probability
        probability = max(
            0.0,
            min(1.0, probability)
        )


        if probability >= 0.5:

            label = "Malignant"
            confidence = probability

        else:

            label = "Benign"
            confidence = 1 - probability


        confidence_percent = round(
            confidence * 100,
            2
        )


        result = {
            "prediction": label,
            "confidence": confidence_percent,
            "inference_time": round(elapsed, 2)
        }


        print(
            f"RESULT: {result}",
            flush=True
        )


        return jsonify(result)


    except Exception as e:

        print(
            "ANALYSIS ERROR:",
            flush=True
        )

        print(
            traceback.format_exc(),
            flush=True
        )


        return jsonify({
            "error": "Analysis failed.",
            "details": str(e)
        }), 500


    finally:

        # Release request memory
        if processed_image is not None:
            del processed_image

        gc.collect()


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image too large. Maximum size is 10MB."
    }), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )

