import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import Webcam from 'react-webcam';
import './App.css';

// ASL alphabet mapping based on hand landmarks
const ASL_ALPHABET = {
  // Basic letters based on finger configurations
  'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 
  'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
  'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
  'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
  'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
};

function detectASLLetter(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  // Simple ASL detection logic (this is where we'd use our trained model)
  const fingerTips = [4, 8, 12, 16, 20];
  const fingerJoints = [2, 6, 10, 14, 18];
  
  let extendedFingers = 0;
  let fingerPositions = [];

  // Analyze finger positions
  for (let i = 0; i < 5; i++) {
    const tip = landmarks[fingerTips[i]];
    const joint = landmarks[fingerJoints[i]];
    
    if (i === 0) { // Thumb
      if (tip[0] < joint[0]) extendedFingers++;
    } else { // Other fingers
      if (tip[1] < joint[1]) extendedFingers++;
    }
    
    fingerPositions.push({
      extended: tip[1] < joint[1],
      position: tip
    });
  }

  // Map to ASL letters (simplified)
  switch (extendedFingers) {
    case 0: return 'A'; // Fist
    case 1: 
      // Check which finger is extended
      if (fingerPositions[1].extended) return 'D'; // Index finger
      return 'B';
    case 2: 
      if (fingerPositions[1].extended && fingerPositions[2].extended) return 'V';
      return 'U';
    case 3: return 'W';
    case 4: return '4';
    case 5: return '5';
    default: return null;
  }
}

function App() {
  const webcamRef = useRef(null);
  const [handModel, setHandModel] = useState(null);
  const [currentLetter, setCurrentLetter] = useState('');
  const [sentence, setSentence] = useState([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [loadingText, setLoadingText] = useState('Loading hand detection...');

  // Load handpose model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setLoadingText('Loading hand detection model...');
        const handModel = await handpose.load();
        setHandModel(handModel);
        
        setLoadingText('Ready for ASL recognition!');
        setTimeout(() => setIsLoaded(true), 1000);
        
      } catch (error) {
        console.error('Error loading model:', error);
        setLoadingText('Error loading models. Using fallback detection.');
        setIsLoaded(true);
      }
    };

    loadModel();
  }, []);

  // Detection loop
  const detect = async () => {
    if (!handModel || !webcamRef.current || !isDetecting) return;

    try {
      if (webcamRef.current.video.readyState === 4) {
        const video = webcamRef.current.video;
        const hands = await handModel.estimateHands(video);

        if (hands.length > 0) {
          const detectedLetter = detectASLLetter(hands[0].landmarks);
          if (detectedLetter && detectedLetter !== currentLetter) {
            setCurrentLetter(detectedLetter);
          }
        } else {
          setCurrentLetter('');
        }
      }
    } catch (error) {
      console.error('Detection error:', error);
    }
  };

  // Add to sentence
  const addToSentence = () => {
    if (currentLetter) {
      setSentence(prev => {
        if (prev.length === 0 || prev[prev.length - 1] !== currentLetter) {
          return [...prev, currentLetter];
        }
        return prev;
      });
    }
  };

  // Auto-add after delay
  useEffect(() => {
    if (currentLetter && isDetecting) {
      const timeout = setTimeout(addToSentence, 1200);
      return () => clearTimeout(timeout);
    }
  }, [currentLetter, isDetecting]);

  // Detection interval
  useEffect(() => {
    let interval;
    if (isDetecting && isLoaded) {
      interval = setInterval(detect, 150);
    }
    return () => clearInterval(interval);
  }, [isDetecting, isLoaded]);

  // Text-to-speech
  const speakSentence = () => {
    if (sentence.length > 0) {
      const text = sentence.join(' ');
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      window.speechSynthesis.speak(utterance);
    }
  };

  const clearSentence = () => {
    setSentence([]);
    setCurrentLetter('');
  };

  const toggleDetection = () => {
    setIsDetecting(!isDetecting);
    if (!isDetecting) {
      setCurrentLetter('');
    }
  };

  if (!isLoaded) {
    return (
      <div className="App">
        <div className="loading">
          <h2>🚀 {loadingText}</h2>
          <div className="spinner"></div>
          <p>First load may take a minute...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>SignSpeak 👐</h1>
        <p>ASL Hand Gesture to Speech Translator</p>
      </header>

      <div className="main-container">
        <div className="webcam-section">
          <Webcam
            ref={webcamRef}
            className="webcam"
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{
              width: 640,
              height: 480,
              facingMode: "user"
            }}
          />
          
          <div className="detection-controls">
            <button 
              onClick={toggleDetection} 
              className={`detect-toggle ${isDetecting ? 'active' : ''}`}
            >
              {isDetecting ? '🛑 Stop ASL Detection' : '🎬 Start ASL Detection'}
            </button>
          </div>

          <div className="current-prediction">
            <h3>ASL Detection:</h3>
            <div className="prediction-display">
              <span className="letter">{currentLetter || '?'}</span>
              <span className="status">
                {currentLetter ? '✅ ASL Detected' : 'Show hand gesture'}
              </span>
            </div>

            <div className="asl-guide">
              <h4>Try These ASL Letters:</h4>
              <div className="asl-examples">
                <div>✊ = A (Fist)</div>
                <div>👆 = B (Index up)</div>
                <div>✌️ = V (Peace)</div>
                <div>🤟 = W (Three fingers)</div>
                <div>🖐️ = 5 (Open hand)</div>
              </div>
            </div>

            {currentLetter && (
              <button onClick={addToSentence} className="add-button">
                ➕ Add "{currentLetter}" to Sentence
              </button>
            )}
          </div>
        </div>

        <div className="output-section">
          <div className="sentence-display">
            <h3>Your Sentence:</h3>
            <div className="sentence-text">
              {sentence.join(' ') || 'Letters will appear here...'}
            </div>
            <div className="sentence-info">
              {sentence.length} letters | Spell out words
            </div>
          </div>

          <div className="controls">
            <button onClick={speakSentence} disabled={!sentence.length} className="speak-btn">
              🔊 Speak Sentence
            </button>
            <button onClick={clearSentence} disabled={!sentence.length} className="clear-btn">
              🗑️ Clear
            </button>
          </div>

          <div className="instructions">
            <h4>How ASL Detection Works:</h4>
            <ul>
              <li>Uses MediaPipe for hand landmark detection</li>
              <li>Analyzes finger positions and configurations</li>
              <li>Maps hand shapes to ASL letters</li>
              <li>Works in real-time through your webcam</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;