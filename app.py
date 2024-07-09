from flask import Flask, request, jsonify
import numpy as np
import face_recognition
from PIL import Image
import io
import os
import cv2

app = Flask(__name__)

signatures_class = np.load('FaceSignatures_db.npy')
X = signatures_class[:, 0: -1].astype('float')
Y = signatures_class[:, -1]

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        image = Image.open(file.stream)
        img = np.array(image)
        img_resize = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find face locations
        facesCurrent = face_recognition.face_locations(img_resize)
        if facesCurrent:
            encodesCurrent = face_recognition.face_encodings(img_resize, facesCurrent)
            images_path = './images'
            matched_images = []

            for file in os.listdir(images_path):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(images_path, file)
                    image_Check = Image.open(path)
                    image_Check_array = np.array(image_Check)
                    image_to_check_array_resized = cv2.cvtColor(image_Check_array, cv2.COLOR_BGR2RGB)

                    faces_to_check = face_recognition.face_locations(image_to_check_array_resized)
                    encodes_to_check = face_recognition.face_encodings(image_to_check_array_resized, faces_to_check)

                    for encode_check in encodes_to_check:
                        matches = face_recognition.compare_faces(encodesCurrent, encode_check)
                        if matches[0]:
                            matched_images.append(path)

            return jsonify({"matched_images": matched_images}), 200

    return jsonify({"error": "No faces found"}), 400

if __name__ == '__main__':
    app.run(debug=True)
