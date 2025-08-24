from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import numpy as np

app = Flask(__name__)

# Configuration
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# Image preprocessing transforms (same as training)
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class EarlyStopping:
    def __init__(self, patience=7, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

def load_model():
    """Load the trained DenseNet121 model"""
    try:
        from torchvision.models import DenseNet121_Weights
        
        print("Creating model architecture...")
        # Load model with same architecture as training
        weights = DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        
        # Freeze feature layers (same as training)
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Replace classifier (same as training)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        
        print(f"Model architecture created with {num_ftrs} features")
        
        # Load trained weights with better error handling
        if os.path.exists(MODEL_PATH):
            try:
                print(f"Loading weights from {MODEL_PATH}")
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                
                # Handle different ways the model might have been saved
                if 'state_dict' in state_dict:
                    # If saved as {'state_dict': model.state_dict()}
                    model.load_state_dict(state_dict['state_dict'])
                elif 'model_state_dict' in state_dict:
                    # If saved as {'model_state_dict': model.state_dict()}
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    # If saved directly as model.state_dict()
                    model.load_state_dict(state_dict)
                
                print(f"Model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Could not load weights from {MODEL_PATH}: {e}")
                print("Using model with ImageNet pre-trained weights only")
        else:
            print(f"Warning: Model file {MODEL_PATH} not found. Using ImageNet pre-trained weights.")
        
        model = model.to(DEVICE)
        model.eval()
        
        # Test the model with dummy input
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"Model test passed. Output shape: {test_output.shape}")
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load model at startup
print("Loading model...")
model = load_model()

if model is None:
    print("CRITICAL: Model failed to load!")
else:
    print("Model loaded successfully!")

def preprocess_image(image):
    """Preprocess the uploaded image for model inference"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply the same transforms as during training
        image_tensor = test_transforms(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise

def predict_covid(image):
    """Make prediction on the preprocessed image"""
    try:
        if model is None:
            raise ValueError("Model is not loaded")
            
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Class 0: Normal, Class 1: COVID-19 positive
            class_names = ['Normal', 'COVID-19 Positive']
            prediction = class_names[predicted.item()]
            confidence_score = confidence.item() * 100
            
            return prediction, confidence_score, probabilities[0].cpu().numpy()
    except Exception as e:
        print(f"Error in predict_covid: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        print(f"Processing file: {file.filename}")
        
        # Read and process the image
        try:
            image = Image.open(io.BytesIO(file.read()))
            print(f"Image loaded: {image.format}, {image.mode}, {image.size}")
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Preprocess image
        try:
            processed_image = preprocess_image(image)
            print("Image preprocessed successfully")
        except Exception as e:
            return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 500
        
        # Make prediction
        try:
            prediction, confidence, probabilities = predict_covid(processed_image)
            print(f"Prediction: {prediction} ({confidence:.2f}%)")
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Convert image to base64 for display
        try:
            image_rgb = image.convert('RGB')
            buffered = io.BytesIO()
            image_rgb.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            return jsonify({'error': f'Image conversion failed: {str(e)}'}), 500
        
        result = {
            'prediction': prediction,
            'confidence': round(float(confidence), 2),
            'probabilities': {
                'Normal': round(float(probabilities[0]) * 100, 2),
                'COVID-19 Positive': round(float(probabilities[1]) * 100, 2)
            },
            'image': img_str
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'device': str(DEVICE),
        'model_loaded': model is not None,
        'model_path_exists': os.path.exists(MODEL_PATH)
    })

@app.route('/debug')
def debug():
    """Debug endpoint to test model functionality"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
        return jsonify({
            'status': 'success',
            'output_shape': list(output.shape),
            'probabilities': [float(p) for p in probabilities[0].cpu().tolist()],
            'device': str(DEVICE),
            'model_type': type(model).__name__
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
