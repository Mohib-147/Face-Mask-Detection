import numpy as np
import pickle

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, learning_rate=0.01):
        
        
        self.learning_rate = learning_rate
        
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size_1))
        self.b1 = np.zeros((1, hidden_size_1))
        
        self.W2 = np.random.uniform(-1, 1, (hidden_size_1, hidden_size_2))
        self.b2 = np.zeros((1, hidden_size_2))
        
        self.W3 = np.random.uniform(-1, 1, (hidden_size_2, output_size))
        self.b3 = np.zeros((1, output_size))
        
        self.cache = {}
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def softmax(self, x):
        """
        Softmax activation for output layer
        Converts outputs to probability distribution
        """
        # Subtract max for numerical stability
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        
        self.cache = {
            'X': X,
            'z1': self.z1, 'a1': self.a1,
            'z2': self.z2, 'a2': self.a2,
            'z3': self.z3, 'a3': self.a3
        }
        
        return self.a3
    
    def compute_loss(self, y_true, y_pred):
        
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return loss
    
    def backward(self, y_true):
        
        batch_size = y_true.shape[0]
        
        X = self.cache['X']
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        a3 = self.cache['a3']
        
        delta3 = a3 - y_true  
        
        dW3 = np.dot(a2.T, delta3) / batch_size
        db3 = np.sum(delta3, axis=0, keepdims=True) / batch_size
        
        delta2 = np.dot(delta3, self.W3.T) * self.sigmoid_derivative(a2)
        
        dW2 = np.dot(a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size
        
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(a1)
        
        dW1 = np.dot(X.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size
        
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def predict(self, X):
    
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        
        output = self.forward(X)
        return output
    
    def train(self, X_train, y_train, epochs, batch_size=32, verbose=True):
        
        losses = []
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                
                loss = self.compute_loss(y_batch, output)
                epoch_loss += loss
                num_batches += 1
                
                self.backward(y_batch)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def evaluate(self, X_test, y_test):
        
        predictions = self.predict(X_test)
        
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            true_labels = np.argmax(y_test, axis=1)
        else:
            true_labels = y_test.flatten()
        
        accuracy = np.mean(predictions == true_labels)
        
        return accuracy
    
    def save(self, filepath):
        
        weights = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'learning_rate': self.learning_rate
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        
        print(f"✓ Model saved to: {filepath}")
    
    def load(self, filepath):
        
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        self.W3 = weights['W3']
        self.b3 = weights['b3']
        self.learning_rate = weights['learning_rate']
        
        print(f"✓ Model loaded from: {filepath}")