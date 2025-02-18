from libs.models.logistic_regression import LogisticRegression
import numpy as np
from libs.math import softmax

class SoftmaxClassifier(LogisticRegression):
    def __init__(self, num_features: int, num_classes: int):

        super().__init__(num_features) 
        self.parameters = np.random.normal(0, 1e-3, (num_features, num_classes))

    def predict(self, X: np.array) -> np.array:
        """
        Compute the softmax probabilities for each sample and each class.
        """
        scores = np.dot(X, self.parameters)
        return softmax(scores) 

    def predict_labels(self, X: np.array) -> np.array:
        """
        Compute the predicted class for each sample.
        """
        scores = self.predict(X)
        preds = np.argmax(scores, axis=1)
        return preds

    @staticmethod
    def likelihood(preds: np.array, y_onehot: np.array) -> float:
        """
        Compute the cross entropy loss from the predicted labels and the true labels.
        """
        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        N = preds.shape[0]
        loss = -np.sum(y_onehot * np.log(preds)) / N
        return loss

    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Update the weights in-place using the gradient.
        """
        self.parameters -= lr * gradient
    
    
    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Compute the gradient of the cross entropy loss with respect to the parameters.
        """
        N = x.shape[0] 
        error = preds - y 
        jacobian = x.T @ error / N
        return jacobian  