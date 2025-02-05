import unittest
from models.model import SYNXModel
import numpy as np

class TestSYNXModel(unittest.TestCase):

    def setUp(self):
        """
        Initialize the SYNXModel before each test.
        """
        self.model = SYNXModel()

    def test_training(self):
        """
        Test the training function.
        """
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        self.model.train(X_train, y_train)
        predictions = self.model.predict(np.array([[7, 8]]))
        self.assertEqual(len(predictions), 1)

    def test_preprocessing(self):
        """
        Test the preprocessing step (scaling the data).
        """
        data = np.array([[1, 2], [3, 4], [5, 6]])
        processed_data = self.model.preprocess_data(data)
        self.assertEqual(processed_data.shape, data.shape)

    def test_predict(self):
        """
        Test the prediction function.
        """
        X_test = np.array([[5, 6]])
        predictions = self.model.predict(X_test)
        self.assertGreater(len(predictions), 0)

if __name__ == "__main__":
    unittest.main()
