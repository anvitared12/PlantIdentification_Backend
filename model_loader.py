import numpy as np
from pathlib import Path
from PIL import Image
import tflite_runtime.interpreter as tflite

def load_keras_model(model_path: str, class_names_path: str):
    model_file = Path(model_path)
    print(f"[model] Looking for model at: {model_file.resolve()}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file.resolve()}")

    interpreter = tflite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])

    print(f"[model] Loaded TFLite model. Input size: {img_size}")

    names_file = Path(class_names_path)
    if names_file.exists():
        class_names = [l.strip() for l in names_file.read_text().splitlines() if l.strip()]
        print(f"[model] Loaded {len(class_names)} class names")
    else:
        class_names = [f"plant_{i}" for i in range(output_details[0]['shape'][-1])]
        print(f"[model] No class_names.txt — using generic labels")

    return interpreter, class_names, img_size


def preprocess(image: Image.Image, img_size: tuple) -> np.ndarray:
    img = image.resize(img_size)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # EfficientNet preprocessing
    return np.expand_dims(arr, axis=0)


def predict_plant(model, class_names: list, image: Image.Image, img_size: tuple) -> dict:
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    arr = preprocess(image, img_size)
    model.set_tensor(input_details[0]['index'], arr)
    model.invoke()

    preds = model.get_tensor(output_details[0]['index'])[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    plant_name = class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"

    print(f"[model] Predicted: {plant_name} ({confidence:.4f})")
    return {"plant_name": plant_name, "confidence": confidence}