
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Needed for session management

# Configure static folder and upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

# Load the pre-trained model
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, 'models', 'model.pth')

# Create a new instance of the pre-trained model with custom modifications
model = models.resnet50(pretrained=False)
class_names = ['landslide', 'nonlandslide']

# Modify the fully connected layer to match the desired output
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1000),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, len(class_names))
)

# Load the state dictionary
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
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
                    # Secure the filename to avoid unwanted characters
                    filename = secure_filename(image_file.filename)

                    # Path to save the uploaded image
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    # Create the uploads directory if it doesn't exist
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        os.makedirs(app.config['UPLOAD_FOLDER'])

                    # Save the uploaded image
                    image_file.save(upload_path)

                    # Optional: Process the image with the model (e.g., prediction)
                    image = Image.open(image_file.stream)
                    image_tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        output = model(image_tensor)
                        _, predicted = torch.max(output.data, 1)
                        prediction = class_names[predicted.item()]

                    # Store the filename and prediction in the session
                    session['image_filename'] = filename
                    session['prediction'] = prediction

                    # Redirect to the result page
                    return redirect(url_for('result'))

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"

    return render_template('index.html', error_message=error_message)

@app.route('/result', methods=['GET'])
def result():
    # Retrieve the filename and prediction from the session
    image_filename = session.get('image_filename', None)
    prediction = session.get('prediction', None)

    # If no filename or prediction, redirect to the index page
    if not image_filename or not prediction:
        return redirect(url_for('index'))

    return render_template('result.html', image_filename=image_filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)







# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import torchvision.models as models
# import torch.nn as nn
# from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

# # Initialize the Flask app
# app = Flask(__name__)
# app.secret_key = 'super_secret_key'  # Needed for session management

# # Configure static folder and upload folder
# app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

# # Load the pre-trained model
# project_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(project_dir, 'models', 'model.pth')

# # Create a new instance of the pre-trained model with custom modifications
# model = models.resnet50(pretrained=False)
# class_names = ['landslide', 'nonlandslide']

# # Modify the fully connected layer to match the desired output
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 1000),
#     nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(1000, 512),
#     nn.ReLU(),
#     nn.Dropout(p=0.4),
#     nn.Linear(512, 128),
#     nn.ReLU(),
#     nn.Dropout(p=0.3),
#     nn.Linear(128, len(class_names))
# )

# # Load the state dictionary
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()  # Set the model to evaluation mode

# # Define the transformation to apply to the input image
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# def classify(model, image_tensor):
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = torch.max(output.data, 1)
#         prediction = class_names[predicted.item()]
#         confidence_score = output.softmax(dim=1)[0][predicted].item()
#     return prediction, confidence_score

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     error_message = None

# #     if request.method == 'POST':
# #         if 'image' not in request.files:
# #             error_message = "No image file received"
# #         else:
# #             image_file = request.files['image']
# #             if image_file.filename == '':
# #                 error_message = "Please select an image file"
# #             else:
# #                 try:
# #                     # Secure the filename to avoid unwanted characters
# #                     filename = secure_filename(image_file.filename)

# #                     # Path to save the uploaded image
# #                     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

# #                     # Create the uploads directory if it doesn't exist
# #                     if not os.path.exists(app.config['UPLOAD_FOLDER']):
# #                         os.makedirs(app.config['UPLOAD_FOLDER'])

# #                     # Save the uploaded image
# #                     image_file.save(upload_path)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     error_message = None

#     if request.method == 'POST':
#         if 'image' not in request.files:
#             error_message = "No image file received"
#         else:
#             image_file = request.files['image']
#             if image_file.filename == '':
#                 error_message = "Please select an image file"
#             else:
#                 try:
#                     # Secure the filename to avoid unwanted characters
#                     filename = secure_filename(image_file.filename)

#                     # Path to save the uploaded image
#                     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#                     # Create the uploads directory if it doesn't exist
#                     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#                         os.makedirs(app.config['UPLOAD_FOLDER'])

#                     # Save the uploaded image
#                     image_file.save(upload_path)

#                     # Process the image with the model (e.g., prediction)
#                     image = Image.open(image_file.stream)
#                     image_tensor = transform(image).unsqueeze(0)
#                     prediction, confidence_score = classify(model, image_tensor)
#                     confidence_percent = round(confidence_score * 100, 2)

#                     # Store the filename, prediction, and confidence in the session
#                     session['image_filename'] = filename
#                     session['prediction'] = prediction
#                     session['confidence'] = confidence_percent

#                     # Redirect to the result page
#                     return redirect(url_for('result'))

#                 except Exception as e:
#                     error_message = f"An error occurred: {str(e)}"

#     return render_template('index.html', error_message=error_message)

# @app.route('/result', methods=['GET'])
# def result():
#     # Retrieve the filename, prediction, and confidence from the session
#     image_filename = session.get('image_filename', None)
#     prediction = session.get('prediction', None)
#     confidence = session.get('confidence', None)

#     # If no filename, prediction, or confidence, redirect to the index page
#     if not image_filename or not prediction or confidence is None:
#         return redirect(url_for('index'))

#     return render_template('result.html', image_filename=image_filename, prediction=prediction, confidence=confidence)

# @app.route('/uploads/<filename>')
# def send_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)