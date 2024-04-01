from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model


# load the deep learning model from model.h5
audio_model = load_model('model.h5')
# predict a warm up call on warm_up.wav
warm_up = audio_model.predict('warm_up.wav')


# For this example, let's assume you have a model named 'model' imported from somewhere




app = Flask(__name__)

# Function to perform inference on the uploaded file
def classify(file):
    # Preprocess the uploaded file (if needed)
    # Example: Convert the uploaded file to an image
    img = Image.open(io.BytesIO(file.read()))
    # Example: Convert image to numpy array
    img_array = np.array(img)
    # Perform inference with your neural network model
    # Example: Call the predict function of your model
    prediction = model.predict(img_array)
    # Example: Return the prediction result
    return prediction

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
            # Perform inference
            result = classify(file)
            return render_template('result.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
