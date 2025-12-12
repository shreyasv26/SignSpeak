SignSpeak – Real-Time Sign Language to Speech Converter

SignSpeak is an AI-powered system that converts live sign language gestures into spoken words using a combination of MediaPipe, TensorFlow LSTM, and OpenCV.
The system captures real-time webcam input, extracts body & hand landmarks, classifies gestures, and speaks the predicted label through an integrated TTS engine.

🚀 Features

Real-time gesture detection via webcam

MediaPipe Holistic for pose + hand landmark extraction

LSTM deep learning model trained on custom gesture sequences

TensorFlow Lite for low-latency prediction

Clean UI with bounding box, FPS, and latency display

Text-to-Speech that speaks predicted gestures

Supports custom gestures (hello, thank you, I love you, yes, no, help, please, cat, eat, fine)

📦 Tech Stack
Machine Learning

TensorFlow / Keras

LSTM Neural Networks

TensorFlow Lite

Computer Vision

MediaPipe Holistic

OpenCV

Other Tools

NumPy

pyttsx3 (Text-to-Speech)

Python

Git/GitHub

🎯 Project Workflow

Data Collection

MediaPipe extracts 258 landmark values per frame

20–30 frames per gesture → saved as .npy sequences

Model Training

LSTM model learns temporal movement patterns

Achieved ~90% accuracy on test set

Model exported to .tflite for real-time inference

Real-Time Inference

Webcam feed processed frame-by-frame

Recent frames form a sequence

TFLite model predicts gesture

UI shows label, confidence, FPS

TTS engine speaks the predicted gesture

▶️ How to Run

conda activate action
python realtime_interface.py


To run with speech output:
python realtime_interface_speaking.py

📊 Model Performance

Accuracy: ~90–94% (varies by class)

Low-latency predictions (~20–30ms)

Smooth real-time inference

Strong performance on dynamic gestures

🗣️ Output Example

When performing a gesture:

UI shows: "hello"

Smooth blue bounding box around head

FPS & latency visible

TTS says: "hello"

💡 Future Improvements

Add more gestures

Support full sentence generation

Add camera calibration

Create a full React or mobile app UI