# COVID-19 Chest X-Ray Analyzer

A Flask web application that uses a trained DenseNet121 model to analyze chest X-ray images for COVID-19 detection.

## Features

- **AI-Powered Analysis**: Uses a pre-trained DenseNet121 model for accurate COVID-19 detection
- **Interactive Web Interface**: Modern, responsive design with drag-and-drop file upload
- **Real-time Results**: Instant analysis with confidence scores and probability distributions
- **Medical-Grade UI**: Professional medical website design with appropriate colors and transitions
- **Image Preprocessing**: Automatic image preprocessing matching the training pipeline

## Setup Instructions

### 1. Install Dependencies

First, install the required Python packages:

```bash
C:/Users/kanak/Desktop/bebo_project/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Model File

Make sure you have your trained model file `best_model.pth` in the project root directory. This should be the model saved during training with the EarlyStopping callback.

### 3. Run the Application

Start the Flask development server:

```bash
C:/Users/kanak/Desktop/bebo_project/.venv/Scripts/python.exe app.py
```

The application will be available at: `http://localhost:5000`

## Model Architecture

The model uses the same architecture as your training setup:

- **Base Model**: DenseNet121 with ImageNet pre-trained weights
- **Frozen Features**: Feature extraction layers are frozen
- **Custom Classifier**: 
  - Linear layer: 1024 → 256 features
  - ReLU activation
  - Dropout (0.4)
  - Output layer: 256 → 2 classes (Normal, COVID-19 Positive)

## Image Preprocessing

Images are preprocessed using the same pipeline as training:

1. Resize to 224x224 pixels
2. Convert to RGB format
3. Normalize with ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Usage

1. **Upload Image**: Click the upload area or drag and drop a chest X-ray image
2. **Analyze**: Click the "Analyze X-Ray" button
3. **View Results**: See the prediction, confidence score, and detailed probabilities

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)

## Medical Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. The results should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## Technical Details

- **Framework**: Flask
- **Deep Learning**: PyTorch
- **Model**: DenseNet121
- **Input Size**: 224x224 RGB images
- **Classes**: 2 (Normal, COVID-19 Positive)
- **Device**: Automatically detects CUDA availability

## File Structure

```
bebo_project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── model_reference.py     # Model architecture reference
├── best_model.pth        # Trained model weights (you need to add this)
├── templates/
│   └── index.html        # Web interface
├── static/               # Static files (CSS, JS, images)
└── README.md            # This file
```

## Troubleshooting

1. **Model not found**: Ensure `best_model.pth` is in the project root directory
2. **CUDA issues**: The app automatically falls back to CPU if CUDA is not available
3. **Import errors**: Make sure all dependencies are installed correctly
4. **Image format errors**: Ensure images are in supported formats (JPEG, PNG)

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Image prediction API
- `GET /health`: Health check endpoint
