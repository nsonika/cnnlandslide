import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure File Uploads (Adjust path as needed)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Load the pre-trained model
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, 'models', 'model.pth')

# Create a new instance of your model
model = models.resnet50(pretrained=False)
class_names = ['landslide', 'nonlandslide']

# Modify the fully connected layer to match your saved architecture
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 1000), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(1000, 512), nn.ReLU(), nn.Dropout(p=0.4),
                         nn.Linear(512, 128), nn.ReLU(), nn.Dropout(p=0.3),
                         nn.Linear(128, len(class_names)))

# Load the state dictionary
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    error_message = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error_message = "No image file received"
        else:
            image_file = request.files['image']
            if image_file.filename == '':
                error_message = "Please select an image file"
            else:
                try:
                    filename = secure_filename(image_file.filename)
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    # Create the uploads directory if it doesn't exist
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        os.makedirs(app.config['UPLOAD_FOLDER'])

                    image = Image.open(image_file.stream)
                    image_tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        output = model(image_tensor)
                        _, predicted = torch.max(output.data, 1)
                        prediction = class_names[predicted.item()]

                    image_file.save(image_path)
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction, image_path=image_path, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
