This project implements a real-time gender detection system using deep learning and computer vision. The system can detect faces in a live video stream from a webcam and predict the gender (male/female) with confidence scores. The project includes:

A CNN model for gender classification
Webcam-based real-time detection
Training scripts with data augmentation
Two different implementation variants

Features:
Real-time gender detection from webcam feed
Deep learning model with ~96% accuracy
Data augmentation for robust training
Two implementation variants:
Basic version (detect_gender_webcam.py)
Enhanced version with better visualization (genderdetection.py)
Training pipeline with visualization of metrics

Prerequisites
Python 3.7+
TensorFlow 2.x
OpenCV
cvlib
matplotlib
scikit-learn
Installation

Clone the repository:
bash
Copy
git clone https://github.com/yourusername/gender-detection.git
cd gender-detection

Install the required packages:
pip install tensorflow opencv-python cvlib matplotlib scikit-learn
Download the dataset and place it in the correct directory (for training)

Usage
Using Pre-trained Model
Run the detection script:
python genderdetection.py
python detect_gender_webcam.py
Press 'Q' to quit the application

Training Your Own Model
Prepare your dataset in the following structure:

Copy
gender_dataset_face/
    man/
        image1.jpg
        image2.jpg
        ...
    woman/
        image1.jpg
        image2.jpg
        ...
Run the training script:
python train.py

The script will:
Train the model
Save the model to gender_detection.model
Generate a training metrics plot (plot.png)

Project Structure
Copy
.
├── detect_gender_webcam.py   # Basic webcam gender detection
├── genderdetection.py        # Enhanced webcam gender detection
├── train.py                 # Model training script
├── gender_detection.model    # Pre-trained model (after training)
├── plot.png                  # Training metrics visualization
└── README.md                 # This file


Model Architecture
The CNN model consists of:
5 Conv2D layers with BatchNormalization and ReLU activation
MaxPooling2D and Dropout layers for regularization
Dense layers for classification
Binary cross-entropy loss with Adam optimizer
Performance

The model achieves:
Training accuracy: ~96%
Validation accuracy: ~94%
Real-time performance at 15-20 FPS (depending on hardware)
Customization

You can adjust:
Training parameters in train.py (epochs, learning rate, batch size)
Model architecture in the build() function
Detection threshold in the webcam scripts
Visualization colors and styles

