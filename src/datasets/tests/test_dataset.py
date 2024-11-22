import unittest

from src.datasets.dataset import Dataset
from src.datasets.preprocessor import DatasetPreprocessor


class TestDataset(unittest.TestCase):
    # prep
    def setUp(self):
        self.german_dataset = Dataset(name="german_binary")
        self.german_preprocessor = DatasetPreprocessor(
            self.german_dataset, standardize_data="minmax", one_hot=True
        )

    def test_dataset(self):
        self.assertEqual(self.german_dataset.name, "german_binary")
