import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import importlib.util

sys.path.insert(0, os.path.dirname(__file__))

spec = importlib.util.spec_from_file_location(
    "neural_network", 
    os.path.join(os.path.dirname(__file__), "utils", "neural_network.py")
)
neural_network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_network_module)
NeuralNetwork = neural_network_module.NeuralNetwork

from config import (
    MODEL1_PATH, MODEL2_PATH, MODEL3_PATH,
    MODEL1_LABELS, MODEL2_LABELS, MODEL3_LABELS,
    DATASET_DIR
)

import cv2

class EvaluationEngine:
    
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.load_models()
    
    def load_models(self):
        print("Loading models...")
        
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
            print("✓ Model 1 loaded")
        
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
            print("✓ Model 2 loaded")
        
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
            print("✓ Model 3 loaded")
    
    def load_test_data(self, model_num):
        test_dir = os.path.join(DATASET_DIR, f'model{model_num}_mask_status', 'test') if model_num == 1 else \
                   os.path.join(DATASET_DIR, f'model{model_num}_mask_colour', 'test') if model_num == 2 else \
                   os.path.join(DATASET_DIR, f'model{model_num}_mask_type', 'test')
        
        labels_map = {label: idx for idx, label in enumerate(
            MODEL1_LABELS if model_num == 1 else
            MODEL2_LABELS if model_num == 2 else
            MODEL3_LABELS
        )}
        
        images = []
        labels = []
        
        for label, idx in labels_map.items():
            label_dir = os.path.join(test_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(label_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_flat = img.flatten().astype('float32') / 255.0
                            images.append(img_flat)
                            labels.append(idx)
        
        return np.array(images), np.array(labels)
    
    def evaluate_model(self, model_num):
        if model_num == 1:
            model = self.model1
            labels = MODEL1_LABELS
        elif model_num == 2:
            model = self.model2
            labels = MODEL2_LABELS
        else:
            model = self.model3
            labels = MODEL3_LABELS
        
        if model is None:
            print(f"Model {model_num} not loaded")
            return
        
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL {model_num}")
        print(f"{'='*80}")
        
        X_test, y_test = self.load_test_data(model_num)
        print(f"\nLoaded {len(X_test)} test images")
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n📊 METRICS:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\n📋 PER-CLASS METRICS:")
        for i, label in enumerate(labels):
            if i < len(cm):
                class_accuracy = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                print(f"  {label:20s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        print(f"\n🔀 CONFUSION MATRIX:")
        print("     ", end="")
        for label in labels:
            print(f"{label:15s}", end="")
        print()
        
        for i, label in enumerate(labels):
            if i < len(cm):
                print(f"{label:20s}", end="")
                for j in range(len(labels)):
                    print(f"{cm[i, j]:15d}", end="")
                print()

def main():
    print("\n" + "="*80)
    print("SCRIPT 09: EVALUATE ALL MODELS")
    print("="*80 + "\n")
    
    engine = EvaluationEngine()
    
    engine.evaluate_model(1)
    engine.evaluate_model(2)
    engine.evaluate_model(3)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
