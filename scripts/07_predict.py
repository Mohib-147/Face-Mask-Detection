import os
import sys
import pickle
import numpy as np
import cv2
import importlib.util

spec = importlib.util.spec_from_file_location(
    "neural_network", 
    os.path.join(os.path.dirname(__file__), "utils", "neural_network.py")
)
neural_network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_network_module)
NeuralNetwork = neural_network_module.NeuralNetwork

from config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, MODEL1_LABELS, MODEL2_LABELS, MODEL3_LABELS

class PredictionEngine:
    
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.load_models()
    
    def load_models(self):
        print("="*80)
        print("LOADING MODELS")
        print("="*80)
        
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
            print("✓ Model 1 (Status) loaded")
        
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
            print("✓ Model 2 (Colour) loaded")
        
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
            print("✓ Model 3 (Type) loaded")
        
        print()
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_flat = img.flatten().astype('float32') / 255.0
        return img_flat.reshape(1, -1)
    
    def predict(self, image_path):
        print("="*80)
        print("MAKING PREDICTIONS")
        print("="*80)
        print(f"\nImage: {image_path}\n")
        
        img_data = self.preprocess_image(image_path)
        
        results = {}
        
        if self.model1:
            pred1 = self.model1.predict(img_data)[0]
            results['status'] = MODEL1_LABELS[pred1]
            print(f"Status:  {results['status']}")
        
        if self.model2:
            pred2 = self.model2.predict(img_data)[0]
            results['colour'] = MODEL2_LABELS[pred2]
            print(f"Colour:  {results['colour']}")
        
        if self.model3:
            pred3 = self.model3.predict(img_data)[0]
            results['type'] = MODEL3_LABELS[pred3]
            print(f"Type:    {results['type']}")
        
        print()
        return results

def main():
    print("\n" + "="*80)
    print("SCRIPT 07: MAKE PREDICTIONS ON NEW IMAGES")
    print("="*80 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage: python 07_predict.py <image_path>")
        print("\nExample: python 07_predict.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    engine = PredictionEngine()
    results = engine.predict(image_path)
    
    print("="*80)
    print("✅ PREDICTION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
