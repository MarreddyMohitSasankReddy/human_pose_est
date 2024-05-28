from flask import Flask, request, render_template, send_from_directory
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the skeleton connections
skeleton = [
    (5, 7), (7, 9),   # Right arm
    (6, 8), (8, 10),  # Left arm
    (11, 13), (13, 15),  # Right leg
    (12, 14), (14, 16),  # Left leg
    (5, 11), (6, 12),  # Torso
    (5, 6), (11, 12)  # Shoulders and hips
]

def draw_keypoints_and_skeleton(image, keypoints, skeleton):
    image = np.array(image)
    for keypoint in keypoints:
        x, y, _ = keypoint
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    for connection in skeleton:
        pt1 = keypoints[connection[0]][:2]
        pt2 = keypoints[connection[1]][:2]
        cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)
    
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image
            image = Image.open(file_path).convert("RGB")
            transform = T.Compose([T.ToTensor()])
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                predictions = model(image_tensor)

            keypoints = predictions[0]['keypoints'][0].cpu().numpy()
            result_image = draw_keypoints_and_skeleton(image, keypoints, skeleton)
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

            return render_template('index.html', original_image=filename, result_image='result_' + filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
