import numpy as np
from sklearn.svm import SVR

class SupportTensorRegressor:
    def __init__(self, C=1.0, epsilon=0.1, max_iter=10):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.u = None
        self.v = None
        self.b = 0

    def fit(self, X, y):
        # X is a 3D tensor: (samples, n1, n2)
        m, n1, n2 = X.shape
        
        # 1. Initialize u as a vector of ones (Step 1 in your paper)
        self.u = np.ones(n1) / np.sqrt(n1) [cite: 73]
        
        for _ in range(self.max_iter):
            # 2. Fix u, solve for v (Step 2)
            # Projecting 2nd-order tensor X_i onto u: x_prime = X_i.T @ u
            X_prime = np.array([X[i].T @ self.u for i in range(m)]) [cite: 75]
            
            svr_v = SVR(kernel='linear', C=self.C, epsilon=self.epsilon)
            svr_v.fit(X_prime, y)
            self.v = svr_v.coef_[0]
            self.b = svr_v.intercept_[0] [cite: 76, 80]
            
            # 3. Fix v, solve for u (Step 3)
            # Projecting X_i onto v: x_tilde = X_i @ v
            X_tilde = np.array([X[i] @ self.v for i in range(m)]) [cite: 52]
            
            svr_u = SVR(kernel='linear', C=self.C, epsilon=self.epsilon)
            svr_u.fit(X_tilde, y)
            self.u = svr_u.coef_[0] [cite: 51, 61]
            
        print("STR Model Trained. Parameters reduced significantly.")

# Example usage with 'Off the Wall' data
X_train = np.random.randn(10, 28, 28) # 10 'images' of 28x28
y_train = np.random.randn(10)

model = SupportTensorRegressor()
model.fit(X_train, y_train)