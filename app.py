import os
import time
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


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

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_skin_lesion_model.keras"
)


model = None
model_load_error = None


print("========================================", flush=True)
print("DERMAI STARTING", flush=True)
print("========================================", flush=True)

print(
    "MODEL PATH:",
    MODEL_PATH,
    flush=True
)

print(
    "MODEL EXISTS:",
    os.path.exists(MODEL_PATH),
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

            "Patches":
                Patches,

            "PatchEncoder":
                PatchEncoder

        }

    )


    print(
        "========================================",
        flush=True
    )

    print(
        "MODEL LOADED SUCCESSFULLY",
        flush=True
    )

    print(
        "MODEL INPUT:",
        model.input_shape,
        flush=True
    )

    print(
        "MODEL OUTPUT:",
        model.output_shape,
        flush=True
    )

    print(
        "========================================",
        flush=True
    )


except Exception as e:

    model_load_error = repr(e)

    print(
        "========================================",
        flush=True
    )

    print(
        "MODEL LOAD ERROR:",
        repr(e),
        flush=True
    )

    print(
        "========================================",
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


    # Convert to RGB

    img = pil_img.convert("RGB")


    print(
        "IMAGE CONVERTED TO RGB",
        flush=True
    )


    # Resize

    img = img.resize(
        (224, 224)
    )


    print(
        "IMAGE RESIZED TO 224x224",
        flush=True
    )


    # NumPy

    arr = np.array(
        img,
        dtype=np.float32
    )


    print(
        "NUMPY ARRAY CREATED:",
        arr.shape,
        arr.dtype,
        flush=True
    )


    # Batch dimension

    arr = np.expand_dims(
        arr,
        axis=0
    )


    print(
        "BATCH CREATED:",
        arr.shape,
        flush=True
    )


    return arr


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    print(
        "HOME PAGE REQUEST",
        flush=True
    )

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

    print(
        "========================================",
        flush=True
    )

    print(
        "ANALYZE REQUEST RECEIVED",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    print(
        "CHECKING MODEL...",
        flush=True
    )


    if model is None:

        print(
            "MODEL IS NONE",
            flush=True
        )


        return jsonify({

            "error":
                "AI model could not be loaded.",

            "details":
                model_load_error

        }), 500


    print(
        "MODEL IS READY",
        flush=True
    )


    # ========================================================
    # CHECK FILE
    # ========================================================

    print(
        "CHECKING UPLOADED FILE...",
        flush=True
    )


    print(
        "FILES RECEIVED:",
        list(request.files.keys()),
        flush=True
    )


    if "image" not in request.files:

        print(
            "NO IMAGE FIELD FOUND",
            flush=True
        )


        return jsonify({

            "error":
                "No image was uploaded."

        }), 400


    file = request.files["image"]


    print(
        "FILE RECEIVED:",
        file.filename,
        flush=True
    )


    if file.filename == "":

        print(
            "EMPTY FILENAME",
            flush=True
        )


        return jsonify({

            "error":
                "No image was selected."

        }), 400


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        print(
            "OPENING IMAGE...",
            flush=True
        )


        image = Image.open(
            file.stream
        )


        print(
            "IMAGE OPENED SUCCESSFULLY",
            flush=True
        )


        print(
            "IMAGE SIZE:",
            image.size,
            flush=True
        )


        print(
            "IMAGE MODE:",
            image.mode,
            flush=True
        )


    except Exception as e:

        print(
            "IMAGE OPEN ERROR:",
            repr(e),
            flush=True
        )


        return jsonify({

            "error":
                "The uploaded file could not be processed.",

            "details":
                str(e)

        }), 400


    # ========================================================
    # PREPROCESS
    # ========================================================

    try:

        processed_image = preprocess_image(
            image
        )


        print(
            "PREPROCESSING SUCCESSFUL",
            flush=True
        )


    except Exception as e:

        print(
            "PREPROCESSING ERROR:",
            repr(e),
            flush=True
        )


        return jsonify({

            "error":
                "Image preprocessing failed.",

            "details":
                str(e)

        }), 400


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    try:

        print(
            "========================================",
            flush=True
        )

        print(
            "STARTING MODEL PREDICTION",
            flush=True
        )

        print(
            "THIS IS WHERE TENSORFLOW RUNS",
            flush=True
        )

        print(
            "========================================",
            flush=True
        )


        start_time = time.time()


        prediction = model.predict(

            processed_image,

            verbose=0

        )


        elapsed_time = (
            time.time()
            -
            start_time
        )


        print(
            "========================================",
            flush=True
        )

        print(
            "MODEL PREDICTION FINISHED",
            flush=True
        )

        print(
            "INFERENCE TIME:",
            round(elapsed_time, 2),
            "seconds",
            flush=True
        )

        print(
            "RAW PREDICTION:",
            prediction,
            flush=True
        )

        print(
            "========================================",
            flush=True
        )


        probability = float(

            np.asarray(
                prediction
            ).flatten()[0]

        )


        print(
            "PROBABILITY:",
            probability,
            flush=True
        )


        # ====================================================
        # INTERPRET OUTPUT
        # ====================================================

        if probability > 0.5:

            label = "Malignant"

            confidence = probability

        else:

            label = "Benign"

            confidence = (
                1 -
                probability
            )


        print(
            "PREDICTION:",
            label,
            flush=True
        )


        print(
            "CONFIDENCE:",
            round(
                confidence * 100,
                2
            ),
            "%",
            flush=True
        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        response = {

            "prediction":
                label,

            "confidence":
                round(
                    confidence * 100,
                    2
                )

        }


        print(
            "RETURNING JSON:",
            response,
            flush=True
        )


        return jsonify(
            response
        )


    except Exception as e:

        print(
            "========================================",
            flush=True
        )

        print(
            "MODEL PREDICTION ERROR",
            flush=True
        )

        print(
            repr(e),
            flush=True
        )

        print(
            "========================================",
            flush=True
        )


        return jsonify({

            "error":
                "The AI model could not analyze this image.",

            "details":
                repr(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if model is None:

        return jsonify({

            "status":
                "error",

            "model_loaded":
                False,

            "error":
                model_load_error

        }), 500


    return jsonify({

        "status":
            "ok",

        "model_loaded":
            True,

        "model_input":
            str(model.input_shape),

        "model_output":
            str(model.output_shape)

    })


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
        "STARTING FLASK ON PORT:",
        port,
        flush=True
    )


    app.run(

        host="0.0.0.0",

        port=port

    )

