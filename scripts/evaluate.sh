#!/bin/bash

# Set the Python environment and install dependencies
echo "Setting up environment..."
pip install -r config/requirements.txt

# Run evaluation on the trained model
echo "Evaluating the model..."
python -c "
from models.model import SYNXModel
model = SYNXModel()
X_test = [[5, 6]]
predictions = model.predict(X_test)
print(f'Predictions: {predictions}')
"

echo "Evaluation completed!"
