from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load TensorFlow model
MODEL = None
try:
    MODEL = tf.keras.models.load_model("../saved_models/1")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Function to read uploaded file as image
def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        print("Error reading image file:", e)
        raise HTTPException(status_code=422, detail="Error processing image file")

# POST endpoint for prediction
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        # Read uploaded file as image
        contents = await file.read()
        print("Uploaded file contents:", contents) 
        
        image = read_file_as_image(contents)
        
        # Preprocess image for model
        img_batch = np.expand_dims(image, 0)
        
        # Perform predictions
        predictions = MODEL.predict(img_batch)
        
        # Get predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        # Return prediction result
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        print("Error during prediction:", e)
        raise HTTPException(status_code=422, detail="Error during prediction")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
