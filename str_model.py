import numpy as np
from sklearn.svm import SVR
from typing import Optional, Tuple

class SupportTensorRegressor:
    """
    Support Tensor Regressor (STR).
    
    This model implements a supervised learning algorithm for tensor-valued inputs.
    It uses an alternating optimization strategy to solve for the rank-1 
    weight tensors u and v, leveraging scikit-learn's SVR for the underlying
    optimization steps.
    
    Attributes:
        C (float): Regularization parameter.
        epsilon (float): Epsilon-tube within which no penalty is associated 
            in the training loss function.
        max_iter (int): Maximum number of iterations for the alternating optimization.
        u (Optional[np.ndarray]): The weight vector for the first dimension.
        v (Optional[np.ndarray]): The weight vector for the second dimension.
        b (float): The bias term.
    """
    
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, max_iter: int = 10):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportTensorRegressor":
        """
        Fit the STR model according to the given training data.
        
        Args:
            X (np.ndarray): Training data of shape (samples, n1, n2).
            y (np.ndarray): Target values of shape (samples,).
            
        Returns:
            self: Returns the instance itself.
        """
        m, n1, n2 = X.shape
        
        # 1. Initialize u as a unit vector of ones
        # Initial guess for the projection onto the first dimension
        self.u = np.ones(n1) / np.sqrt(n1)
        
        for iteration in range(self.max_iter):
            # 2. Fix u, solve for v
            # Projecting 2nd-order tensor X_i onto u: x_prime = X_i.T @ u
            X_prime = np.array([X[i].T @ self.u for i in range(m)])
            
            svr_v = SVR(kernel='linear', C=self.C, epsilon=self.epsilon)
            svr_v.fit(X_prime, y)
            self.v = svr_v.coef_[0]
            self.b = svr_v.intercept_[0]
            
            # 3. Fix v, solve for u
            # Projecting X_i onto v: x_tilde = X_i @ v
            X_tilde = np.array([X[i] @ self.v for i in range(m)])
            
            svr_u = SVR(kernel='linear', C=self.C, epsilon=self.epsilon)
            svr_u.fit(X_tilde, y)
            self.u = svr_u.coef_[0]
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression values for X.
        
        Args:
            X (np.ndarray): Input data of shape (samples, n1, n2).
            
        Returns:
            y_pred (np.ndarray): Predicted values.
        """
        if self.u is None or self.v is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        # Prediction: y = u^T * X * v + b
        return np.array([self.u @ X[i] @ self.v for i in range(X.shape[0])]) + self.b

if __name__ == "__main__":
    # Example usage
    X_train = np.random.randn(10, 28, 28)
    y_train = np.random.randn(10)

    model = SupportTensorRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print(f"Predictions: {predictions}")