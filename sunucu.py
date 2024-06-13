from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
import base64
import cv2

# Flask uygulamasını oluştur
app = Flask(__name__)

# Modeli yükle
model = load_model('CanDetect.keras')

@app.route('/', methods=['GET'])
def index():
    return render_template('arayuz.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Kullanıcıdan yüklenen dosyayı al
    img_file = request.files.get('file')
    if img_file:
        # Resmi oku ve işle
        img_bytes = BytesIO(img_file.read()) # Dosyayı BytesIO nesnesine dönüştür
        img = image.load_img(img_bytes, target_size=(50, 50))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Modelle tahmin yap
        prediction = model.predict(img_array)

        # Sonucu belirle
        result = 'Kanserli' if prediction[0][0] > 0.5 else 'Kanserli Değil'

        # Sonucu JSON olarak döndür
        return jsonify({'sonuc': result})

    return jsonify({'hata': 'Dosya yüklenemedi'})

@app.route('/arayuz2')
def arayuz2():
    return render_template('arayuz2.html')

@app.route('/filter1', methods=['POST'])
def filter1_image():
    # Get the uploaded file
    img_file = request.files.get('file')
    if img_file:
        # Read and process the image
        img_bytes = BytesIO(img_file.read())  # Convert the file to a BytesIO object
        img = Image.open(img_bytes)
        img = np.array(img)  # Convert to NumPy array

        # Convert image to grayscale if it is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

        # Define a custom Laplace kernel
        laplace_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        # Apply the custom Laplace filter using the defined kernel
        laplacian = cv2.filter2D(blurred_img, cv2.CV_64F, laplace_kernel)

        # Normalize the result
        laplacian = cv2.convertScaleAbs(laplacian)

        # Combine original and Laplacian images to enhance edges
        combined_img = cv2.addWeighted(img, 0.7, laplacian, 0.3, 0)

        # Convert the result back to an image
        filtered_img = Image.fromarray(combined_img)

        # Convert the filtered image back to bytes
        img_byte_arr = BytesIO()
        filtered_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Convert the bytes to a base64 string
        img_base64 = base64.b64encode(img_byte_arr).decode()

        # Return the base64 string as JSON
        return jsonify({'image': img_base64})

    return jsonify({'error': 'No file uploaded'})

@app.route('/filter2', methods=['POST'])
def filter2_image():
    # Get the uploaded file
    img_file = request.files.get('file')
    if img_file:
        # Read and process the image
        img_bytes = BytesIO(img_file.read())  # Convert the file to a BytesIO object
        img = Image.open(img_bytes)

        # Apply a filter to the image
        embossed_img = img.filter(ImageFilter.EMBOSS)

        # Convert the filtered image back to bytes
        img_byte_arr = BytesIO()
        embossed_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Convert the bytes to a base64 string
        img_base64 = base64.b64encode(img_byte_arr).decode()

        # Return the base64 string as JSON
        return jsonify({'image': img_base64})

    return jsonify({'error': 'No file uploaded'})


@app.route('/filter3', methods=['POST'])
def filter3_image():
    # Get the uploaded file
    img_file = request.files.get('file')
    if img_file:
        # Read and process the image
        img_bytes = BytesIO(img_file.read())  # Convert the file to a BytesIO object
        img = Image.open(img_bytes)

        # Apply a filter to the image
        contoured_img = img.filter(ImageFilter.CONTOUR)

        # Convert the filtered image back to bytes
        img_byte_arr = BytesIO()
        contoured_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Convert the bytes to a base64 string
        img_base64 = base64.b64encode(img_byte_arr).decode()

        # Return the base64 string as JSON
        return jsonify({'image': img_base64})

    return jsonify({'error': 'No file uploaded'})
# Uygulamayı çalıştır
if __name__ == '__main__':
    app.run(debug=True, threaded=False)



