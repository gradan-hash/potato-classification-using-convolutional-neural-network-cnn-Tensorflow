from fastapi import FastAPI, File, UploadFile
from fastapi.logger import logger
import logging
from uvicorn import run
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi.middleware.cors import CORSMiddleware
# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

origins = [
    "http://localhost:5173",  # No trailing slash
    "http://localhost:8000",  # If your FastAPI runs here too
    "http://localhost",       # Covering general localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("C:/xampp/htdocs/potato classification/saved_model/1")

# Define the path to the saved model
MODEL_PATH = "C:/xampp/htdocs/potato classification/saved_model/1"

# Create a new Keras model with the TFSMLayer
MODEL = keras.Sequential([
    layers.InputLayer(shape=(256, 256, 3)),
    layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
])

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        # Get predictions from the model
        predictions = MODEL.predict(img_batch)
        logger.info(f"Model output: {predictions}")

        # Extract predictions using the correct key from the dictionary
        predictions_array = predictions['dense_4']

        # Determine the class with the highest probability
        predicted_class_index = np.argmax(predictions_array)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions_array)

        logger.info(f"predicted_class: {predicted_class} confidence {confidence} ")
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    run(app, host='localhost', port=8000)
