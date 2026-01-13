from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained MNIST model
model = load_model('D:/codes/CNN/API/mnist_cnn_model.h5')

@app.route('/predict-digit', methods=['POST'])
def predict_digit():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        # Convert image to grayscale and resize to 28x28
        image = Image.open(file).convert('L')
        image = image.resize((28, 28))

        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Run prediction
        prediction = model.predict(image_array)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'digit': digit,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)