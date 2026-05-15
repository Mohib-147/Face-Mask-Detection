import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import importlib.util

spec = importlib.util.spec_from_file_location(
    "neural_network", 
    os.path.join(os.path.dirname(__file__), "utils", "neural_network.py")
)
neural_network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_network_module)
NeuralNetwork = neural_network_module.NeuralNetwork

class Model1Trainer:
    
    def __init__(self):
        self.model = None
        self.label_map = {
            0: 'No Mask',
            1: 'Proper Mask',
            2: 'Improper Mask'
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def load_images_from_folder(self, folder_path):
        
        images = []
        labels = []
        
        print(f"  Loading from: {folder_path}")
        
        categories = sorted(os.listdir(folder_path))
        
        for category in categories:
            category_path = os.path.join(folder_path, category)
            
            if not os.path.isdir(category_path):
                continue
            
            label = self.reverse_label_map[category]
            
            valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            image_files = [f for f in os.listdir(category_path) if f.endswith(valid_extensions)]
            
            print(f"    {category}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(category_path, img_file)
                
                try:
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    img_flat = img.flatten().astype('float32') / 255.0
                    
                    images.append(img_flat)
                    labels.append(label)
                
                except Exception as e:
                    print(f"    ✗ Error loading {img_file}: {e}")
        
        return np.array(images), np.array(labels)
    
    def one_hot_encode(self, y, num_classes):
        """Convert class labels to one-hot encoding"""
        
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        
        return one_hot
    
    def train(self, X_train, y_train):
        """Train the neural network"""
        
        print("\n" + "=" * 80)
        print("TRAINING MODEL 1 - MASK STATUS CLASSIFIER")
        print("=" * 80)
        
        input_size = X_train.shape[1]
        output_size = len(self.label_map)
        
        print(f"\nNetwork Architecture:")
        print(f"  Input Layer:    {input_size} neurons (224×224×3 flattened)")
        print(f"  Hidden Layer 1: 512 neurons (Sigmoid activation)")
        print(f"  Hidden Layer 2: 256 neurons (Sigmoid activation)")
        print(f"  Output Layer:   {output_size} neurons (Softmax activation)")
        print(f"\nTraining set size: {X_train.shape[0]} images")
        print(f"Classes: {[self.label_map[i] for i in range(output_size)]}")
        
        self.model = NeuralNetwork(input_size=input_size,hidden_size_1=512,hidden_size_2=256,
            output_size=output_size,
            learning_rate=0.001
        )
        
        # One-hot encode labels
        y_train_encoded = self.one_hot_encode(y_train, output_size)
        
        print(f"\nTraining Parameters:")
        print(f"  Epochs: 100")
        print(f"  Batch Size: 32")
        print(f"  Learning Rate: 0.001")
        print(f"  Optimizer: Gradient Descent")
        
        print("\nTraining in progress...")
        
        # Train
        losses = self.model.train(
            X_train, y_train_encoded,
            epochs=100,
            batch_size=32,
            verbose=True
        )
        
        print("\n✅ Training complete!")
        
        return losses
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        
        print("\n" + "=" * 80)
        print("EVALUATING MODEL 1")
        print("=" * 80)
        print(f"\nTest set size: {X_test.shape[0]} images")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"\n📊 METRICS:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\n📋 PER-CLASS METRICS:")
        for i, label in self.label_map.items():
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"  {label:20s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        # Confusion matrix
        print(f"\n🔀 CONFUSION MATRIX:")
        print("     ", end="")
        for label in self.label_map.values():
            print(f"{label:15s}", end="")
        print()
        
        for i, label in self.label_map.items():
            print(f"{label:20s}", end="")
            for j in range(len(self.label_map)):
                print(f"{cm[i, j]:15d}", end="")
            print()
        
        # Save metrics
        self.save_metrics(accuracy, precision, recall, f1, cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        return accuracy, precision, recall, f1, cm
    
    def save_metrics(self, accuracy, precision, recall, f1, cm):
        """Save metrics to file"""
        
        metrics_file = 'results/model1_metrics.txt'
        
        with open(metrics_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL 1 - MASK STATUS CLASSIFIER (FROM SCRATCH)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("NETWORK ARCHITECTURE:\n")
            f.write("  Input Layer:    150528 neurons (224×224×3 flattened)\n")
            f.write("  Hidden Layer 1: 512 neurons (Sigmoid)\n")
            f.write("  Hidden Layer 2: 256 neurons (Sigmoid)\n")
            f.write("  Output Layer:   3 neurons (Softmax)\n")
            f.write("  Loss Function:  Cross-Entropy\n")
            f.write("  Optimizer:      Gradient Descent\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n\n")
            
            f.write("PER-CLASS METRICS:\n")
            for i, label in self.label_map.items():
                class_accuracy = cm[i, i] / cm[i].sum()
                f.write(f"  {label:20s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)\n")
            
            f.write("\nCONFUSION MATRIX:\n")
            f.write("     ")
            for label in self.label_map.values():
                f.write(f"{label:15s}")
            f.write("\n")
            
            for i, label in self.label_map.items():
                f.write(f"{label:20s}")
                for j in range(len(self.label_map)):
                    f.write(f"{cm[i, j]:15d}")
                f.write("\n")
        
        print(f"\n✓ Metrics saved to: {metrics_file}")
    
    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.label_map.values()),
            yticklabels=list(self.label_map.values()),
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Model 1 - Mask Status Classifier\nConfusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path = 'results/confusion_matrices/model1_confusion_matrix.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {output_path}")
        plt.close()
    
    def save_model(self, output_path='models/model1_mask_status.pkl'):
        """Save trained model"""
        
        self.model.save(output_path)

def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("SCRIPT 05: TRAIN MODEL 1 - MASK STATUS CLASSIFIER (FROM SCRATCH)")
    print("=" * 80)
    print("\nArchitecture: 3-Layer Feedforward Neural Network")
    print("Activation: Sigmoid (hidden), Softmax (output)")
    print("Loss: Cross-Entropy")
    print("Optimizer: Gradient Descent with Backpropagation")
    
    try:
        trainer = Model1Trainer()
        
        # Load data
        print("\n[STEP 1] Loading Training Data")
        print("-" * 80)
        X_train, y_train = trainer.load_images_from_folder('dataset/model1_mask_status/train')
        
        if X_train.shape[0] == 0:
            print("❌ No training images found!")
            sys.exit(1)
        
        print(f"✓ Loaded {X_train.shape[0]} training images")
        
        print("\n[STEP 2] Loading Test Data")
        print("-" * 80)
        X_test, y_test = trainer.load_images_from_folder('dataset/model1_mask_status/test')
        
        if X_test.shape[0] == 0:
            print("❌ No test images found!")
            sys.exit(1)
        
        print(f"✓ Loaded {X_test.shape[0]} test images")
        
        # Train
        print("\n[STEP 3] Training Model")
        print("-" * 80)
        losses = trainer.train(X_train, y_train)
        
        # Evaluate
        print("\n[STEP 4] Evaluating Model")
        print("-" * 80)
        accuracy, precision, recall, f1, cm = trainer.evaluate(X_test, y_test)
        
        # Save
        print("\n[STEP 5] Saving Model")
        print("-" * 80)
        trainer.save_model()
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ MODEL 1 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\nBest Metrics:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\n💾 Model saved to: models/model1_mask_status.pkl")
        print(f"📊 Metrics saved to: results/model1_metrics.txt")
        print(f"\n💡 Next step: Run 06_train_model2_type.py")
        print("=" * 80)
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()