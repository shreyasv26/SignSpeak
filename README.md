# SignSpeak: Real-Time Sign Language to Speech Converter

**SignSpeak** is an AI-powered system that converts live sign language gestures into spoken words using a combination of **MediaPipe**, **TensorFlow LSTM**, and **OpenCV**. The system captures real-time webcam input, extracts body and hand landmarks, classifies gestures, and speaks the predicted label through an integrated **TTS engine**.

## Key Features
* **Real-time Gesture Detection**: High-speed processing via webcam input.
* **MediaPipe Holistic**: Advanced extraction of pose and hand landmarks for precision.
* **LSTM Deep Learning**: A model trained on custom gesture sequences to understand temporal patterns.
* **TensorFlow Lite**: Optimized for low-latency, real-time prediction (~20–30ms).
* **Text-to-Speech (TTS)**: Integrated engine that speaks predicted gestures aloud.
* **Gesture Support**: Recognizes "Hello," "Thank You," "I Love You," "Yes," "No," "Help," "Please," "Cat," "Eat," and "Fine."

## Tech Stack
### **Machine Learning**
* TensorFlow / Keras (LSTM Neural Networks)
* TensorFlow Lite

### **Computer Vision**
* MediaPipe Holistic
* OpenCV

### **Other Tools**
* NumPy
* pyttsx3 (Text-to-Speech)
* Python 3.x

## Project Workflow
1. **Data Collection**: MediaPipe extracts 258 landmark values per frame. Sequences are saved as `.npy` files.
2. **Model Training**: An LSTM model learns movement patterns over time, achieving ~90% accuracy.
3. **Real-Time Inference**: Webcam feed is processed frame-by-frame. The TFLite model predicts the gesture, and the UI displays the label and confidence.

## How to Run
Activate your environment and run the interface:
```bash
conda activate action
python realtime_interface.py
```

To run with active speech output:
```bash
python realtime_interface_speaking.py
```

## Model Performance
Accuracy: ~90–94% depending on the specific gesture class.
Latency: Smooth real-time inference with 20–30ms prediction speed.
Robustness: Strong performance on dynamic, moving gestures.

## Output Example
When performing a gesture:
UI shows: "hello"
Smooth blue bounding box around head
FPS & latency visible
TTS says: "hello"
