# Support Tensor Regression (STR)

A lightweight Python implementation of Support Tensor Regression, leveraging `scikit-learn`'s `SVR` for iterative optimization of tensor projections.

## Installation

To use this package, clone the repository and install it locally:

```bash
git clone https://github.com/joshdelon03-hash/support-tensor-regression.git
cd support-tensor-regression
pip install .
```

## Usage

```python
import numpy as np
from str_model import SupportTensorRegressor

# Create dummy data
X_train = np.random.randn(10, 28, 28)
y_train = np.random.randn(10)

# Initialize and fit
model = SupportTensorRegressor()
model.fit(X_train, y_train)
```

## Features
- Iterative alternating optimization.
- Built on top of `scikit-learn` for reliability.
- Significant parameter reduction for tensor-valued inputs.
