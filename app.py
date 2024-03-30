import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torchvision.models as models
import torch.nn as nn  # Add this import statement
from flask import Flask, render_template, request

app = Flask(__name__)

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

# Define the class names
class_names = ['landslide', 'nonlandslide']

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Debug statements
        print(request.files)
        if 'image' not in request.files:
            print("No image file received")
        else:
            image_file = request.files['image']
            print(f"Image file received: {image_file.filename}")

        # Open the image file using PIL
        image = Image.open(image_file.stream)

        # Transform the image
        image_tensor = transform(image).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            prediction = class_names[predicted.item()]

        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
