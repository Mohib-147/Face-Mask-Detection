import os
import cv2
import numpy as np
import pickle
import importlib.util

spec = importlib.util.spec_from_file_location(
    "neural_network", 
    os.path.join(os.path.dirname(__file__), "utils", "neural_network.py")
)
neural_network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_network_module)
NeuralNetwork = neural_network_module.NeuralNetwork

from config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, MODEL1_LABELS, MODEL2_LABELS, MODEL3_LABELS

class InferenceEngine:
    
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.load_models()
    
    def load_models(self):
        if os.path.exists(MODEL1_PATH):
            with open(MODEL1_PATH, 'rb') as f:
                weights1 = pickle.load(f)
            self.model1 = NeuralNetwork(150528, 512, 256, 3)
            self.model1.W1 = weights1['W1']
            self.model1.b1 = weights1['b1']
            self.model1.W2 = weights1['W2']
            self.model1.b2 = weights1['b2']
            self.model1.W3 = weights1['W3']
            self.model1.b3 = weights1['b3']
        
        if os.path.exists(MODEL2_PATH):
            with open(MODEL2_PATH, 'rb') as f:
                weights2 = pickle.load(f)
            self.model2 = NeuralNetwork(150528, 512, 256, 6)
            self.model2.W1 = weights2['W1']
            self.model2.b1 = weights2['b1']
            self.model2.W2 = weights2['W2']
            self.model2.b2 = weights2['b2']
            self.model2.W3 = weights2['W3']
            self.model2.b3 = weights2['b3']
        
        if os.path.exists(MODEL3_PATH):
            with open(MODEL3_PATH, 'rb') as f:
                weights3 = pickle.load(f)
            self.model3 = NeuralNetwork(150528, 512, 256, 3)
            self.model3.W1 = weights3['W1']
            self.model3.b1 = weights3['b1']
            self.model3.W2 = weights3['W2']
            self.model3.b2 = weights3['b2']
            self.model3.W3 = weights3['W3']
            self.model3.b3 = weights3['b3']
    
    def preprocess_frame(self, frame):
        frame_resized = cv2.resize(frame, (224, 224))
        frame_flat = frame_resized.flatten().astype('float32') / 255.0
        return frame_flat.reshape(1, -1)
    
    def predict_frame(self, frame):
        img_data = self.preprocess_frame(frame)
        
        results = {}
        
        if self.model1:
            pred1 = self.model1.predict(img_data)[0]
            results['status'] = MODEL1_LABELS[pred1]
        
        if self.model2:
            pred2 = self.model2.predict(img_data)[0]
            results['colour'] = MODEL2_LABELS[pred2]
        
        if self.model3:
            pred3 = self.model3.predict(img_data)[0]
            results['type'] = MODEL3_LABELS[pred3]
        
        return results
    
    def draw_predictions(self, frame, results):
        h, w = frame.shape[:2]
        
        text = f"Status: {results.get('status', 'N/A')} | Colour: {results.get('colour', 'N/A')} | Type: {results.get('type', 'N/A')}"
        
        cv2.rectangle(frame, (10, 10), (w-10, 60), (0, 255, 0), 2)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run_webcam(self):
        print("="*80)
        print("STARTING WEBCAM INFERENCE")
        print("="*80)
        print("\nPress 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
            
            results = self.predict_frame(frame)
            frame = self.draw_predictions(frame, results)
            
            cv2.imshow('Face Mask Detection - Inference', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Webcam inference stopped")
    
    def run_video(self, video_path):
        print("="*80)
        print("STARTING VIDEO INFERENCE")
        print("="*80)
        print(f"\nVideo: {video_path}\n")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            results = self.predict_frame(frame)
            frame = self.draw_predictions(frame, results)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"\n✓ Output video saved to: {output_path}")
        print("✅ Video inference complete")

def main():
    import sys
    
    print("\n" + "="*80)
    print("SCRIPT 08: REAL-TIME INFERENCE")
    print("="*80 + "\n")
    
    engine = InferenceEngine()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'webcam':
            engine.run_webcam()
        else:
            engine.run_video(sys.argv[1])
    else:
        engine.run_webcam()

if __name__ == "__main__":
    main()
