import unittest
import pandas as pd

class TestData(unittest.TestCase):

    def setUp(self):
        """
        Setup for data-related tests.
        """
        self.dataset = pd.read_csv('data/dataset.csv')

    def test_data_loading(self):
        """
        Test if the dataset is loaded properly.
        """
        self.assertIsNotNone(self.dataset)
        self.assertTrue('feature1' in self.dataset.columns)
        self.assertTrue('target' in self.dataset.columns)

    def test_data_preprocessing(self):
        """
        Test data preprocessing steps, such as handling missing values or scaling.
        """
        preprocessed_data = pd.read_csv('data/preprocessed_data.csv')
        self.assertEqual(len(preprocessed_data), len(self.dataset))
        self.assertTrue((preprocessed_data['feature1'].values < 0).all())

if __name__ == "__main__":
    unittest.main()
