import os
import cv2
import numpy as np
import pickle

class ImagePreprocessor:
    
    @staticmethod
    def load_and_preprocess(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_flat = img.flatten().astype('float32') / 255.0
        return img_flat.reshape(1, -1)
    
    @staticmethod
    def load_batch(image_paths):
        batch = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                img_flat = img.flatten().astype('float32') / 255.0
                batch.append(img_flat)
        return np.array(batch)

class ModelLoader:
    
    @staticmethod
    def load_model(model_path, model_class):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
        
        return weights
    
    @staticmethod
    def save_model(model, output_path):
        weights = {
            'W1': model.W1,
            'b1': model.b1,
            'W2': model.W2,
            'b2': model.b2,
            'W3': model.W3,
            'b3': model.b3
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(weights, f)
