import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json
import traceback
import os

# === Configuration ===
IMAGE_SIZE = (256, 256)
CLASSIFIER_MODEL_PATH = "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Py. Code/classifier_model.keras"
REGRESSION_MODEL_PATHS = {"Ischemic": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Py. Code/regression_ischemic_model.keras",
                            "Haemorrhagic": "A:/Siena/Bio-Tech Instrumentation/Stroke recovery time perdiction/Py. Code/regression_haemorrhagic_model.keras"
                         }

# === Load Models Once ===
try:
    classifier = load_model(CLASSIFIER_MODEL_PATH)
    regression_models = { stroke_type: load_model(model_path)
                          for stroke_type, model_path in REGRESSION_MODEL_PATHS.items()
                        }
except Exception as e:
    raise RuntimeError("Model loading failed: " + str(e))

# === Prediction Function ===
def predict_stroke_from_path(image_path):
    try:
        if not isinstance(image_path, str):
            raise TypeError("Image path must be a string")

        if "dummy" in image_path.lower():
            return json.dumps({"stroke_type": "Warmup", "recovery_months": 0.0}, ensure_ascii=False)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # == Load and preprocess image ==
        img = Image.open(image_path)
        img_array = np.array(img).astype(np.float32) / 255.0

        if img_array.shape != (256, 256):
            raise ValueError(f"Expected image shape (256, 256), got {img_array.shape}")

        img_array = img_array.reshape(1, 256, 256, 1)

        # == Predict class ==
        class_probs = classifier.predict(img_array)
        class_index = int(np.argmax(class_probs))
        label_map_rev = {0: "Normal", 1: "Ischemic", 2: "Haemorrhagic"}
        stroke_type = label_map_rev.get(class_index, "Unknown")

        # == Predict recovery time ==
        if stroke_type in regression_models:
            recovery_time = float(regression_models[stroke_type].predict(img_array)[0][0])
            recovery_time = round(max(recovery_time, 0.0), 2)
        else:
            recovery_time = 0.0

        return json.dumps({"stroke_type": stroke_type, "recovery_months": recovery_time}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"stroke_type": "Error", "recovery_months": 0.0, "error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)

# === LabVIEW Callable Function ===
def labview_predict_from_path(image_path: str) -> str:
    return predict_stroke_from_path(image_path)