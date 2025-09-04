import os 
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tensorflow as tf
from tensorflow.keras.models import load_model

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# Limit file size to 1MB
app.config["MAX_CONTENT_LENGTH"] = 1*1024*1024

# Load model
model = load_model("mnist_cnn.h5")

# Setup a limiter so that each IP can only hit /predict 10 times per minute
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per hour"]  # global default
)

# Preprocess images into input for the model 
# Format: (1, 28, 28, 1) tensors, and white digit on black background
def preprocess_image(file_bytes):
        
    img = Image.open(io.BytesIO(file_bytes)).convert("L") # Black and white
    img = img.resize((28,28)) # MNIST size
    arr = np.array(img).astype("float32") / 255.0 # Convert to array, normalize
    # Invert predominantly light images for a bright digit in a dark background
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = arr.reshape(1, 28, 28, 1)
    return arr
      
# Test route
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "MNIST CNN is live."})

# Set route 
@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
# Takes a handwritten digit image and predicts what digit it is.
def predict():
    # Get the image
    if "file" in request.files:
        file_storage = request.files["file"]
        file_bytes = file_storage.read()
    # Handle case where no image was provided
    else: 
        file_bytes = request.get_data()
        if not file_bytes:
            return jsonify({"error": "No image provided"}), 400
    # Preprocess image with handling for edge cases    
    try: 
        x = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    
    # Get probabilities for each digit
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs)) 
    
    # Take the top 3 most probable digits and probabilities
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = \
    [{"digit": int(i),  "prob": f"{probs[i] * 100:.2f}%"} for i in top3_idx]
    
    return jsonify({
        "prediction": pred,
        "confidence": f"{probs[pred] * 100:.2f}%",
        "top3": top3
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True) # TODO:Remove debug = True 