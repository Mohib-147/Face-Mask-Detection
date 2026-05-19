import os
import sys
import pickle
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils.neural_network import NeuralNetwork
from scripts.config import CONFIG

class PredictionEngine:
    
    def __init__(self):
        self.models = {}
        self.label_maps = {
            1: {0: 'no_mask', 1: 'proper_mask', 2: 'improper_mask'},
            2: {0: 'black', 1: 'blue', 2: 'green', 3: 'red', 4: 'white', 5: 'yellow'},
            3: {0: 'cloth', 1: 'medical', 2: 'n95'}
        }
        self.load_models()
    
    def load_models(self):
        print("Loading models...")
        
        for model_num in [1, 2, 3]:
            model_path = CONFIG['MODELS_PATH'][f'model{model_num}']
            
            if not os.path.exists(model_path):
                print(f"✗ Model {model_num} not found: {model_path}")
                continue
            
            with open(model_path, 'rb') as f:
                weights = pickle.load(f)
            
            if model_num == 1:
                model = NeuralNetwork(150528, 512, 256, 3, learning_rate=0.01)
            elif model_num == 2:
                model = NeuralNetwork(150528, 512, 256, 6, learning_rate=0.01)
            else:
                model = NeuralNetwork(150528, 512, 256, 3, learning_rate=0.01)
            
            model.W1 = weights['W1']
            model.b1 = weights['b1']
            model.W2 = weights['W2']
            model.b2 = weights['b2']
            model.W3 = weights['W3']
            model.b3 = weights['b3']
            
            self.models[model_num] = model
            print(f"✓ Model {model_num} loaded")
        
        print()
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        img_resized = cv2.resize(img, (224, 224))
        img_flat = img_resized.flatten().astype('float32') / 255.0
        
        return img_flat.reshape(1, -1), img_resized
    
    def predict(self, image_path):
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        print("="*80)
        print(f"IMAGE: {image_path}")
        print("="*80)
        
        result = self.preprocess_image(image_path)
        if result is None:
            print("✗ Failed to load image")
            return
        
        X, img_display = result
        
        print("\n📊 PREDICTIONS:\n")
        
        if 1 in self.models:
            output = self.models[1].forward(X)
            pred_idx = np.argmax(output[0])
            confidence = output[0][pred_idx] * 100
            pred_label = self.label_maps[1][pred_idx]
            print(f"  Status:  {pred_label.upper():20s} ({confidence:.2f}%)")
        
        if 2 in self.models:
            output = self.models[2].forward(X)
            pred_idx = np.argmax(output[0])
            confidence = output[0][pred_idx] * 100
            pred_label = self.label_maps[2][pred_idx]
            print(f"  Colour:  {pred_label.upper():20s} ({confidence:.2f}%)")
        
        if 3 in self.models:
            output = self.models[3].forward(X)
            pred_idx = np.argmax(output[0])
            confidence = output[0][pred_idx] * 100
            pred_label = self.label_maps[3][pred_idx]
            print(f"  Type:    {pred_label.upper():20s} ({confidence:.2f}%)")
        
        print("\n" + "="*80 + "\n")

def select_image():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    
    return file_path

def main():
    print("\n" + "="*80)
    print("SCRIPT 07: PREDICT - IMAGE CLASSIFICATION")
    print("="*80 + "\n")
    
    image_path = select_image()
    
    if not image_path:
        print("No image selected.")
        return
    
    engine = PredictionEngine()
    engine.predict(image_path)

if __name__ == "__main__":
    main()
