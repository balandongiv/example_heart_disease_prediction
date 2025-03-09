import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import pandas as pd

# Import the functions to test from the correct path.
# These functions are responsible for downloading and loading the dataset.
from heart_disease_prediction.data.data_loader import (
    download_dataset,
    check_and_download_data,
    load_data
)

# Constants used for testing purposes.
# We override the actual DATA_PATH, DOWNLOAD_PATH, and KAGGLE_DATA_URL with test values.
TEST_DATA_PATH = "some/path/to/test_heart_data.csv"
TEST_DOWNLOAD_PATH = "some/path/to/download_dir"
TEST_KAGGLE_URL = "https://www.kaggle.com/datasets/mahatiratusher/mockdownload"

class TestDataLoader(unittest.TestCase):

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.DOWNLOAD_PATH", TEST_DOWNLOAD_PATH)
    @patch("heart_disease_prediction.data.data_loader.KAGGLE_DATA_URL", TEST_KAGGLE_URL)
    @patch("os.path.exists")
    def test_check_and_download_data_exists(self, mock_exists):
        """
        Test that if the dataset already exists, the download_dataset function is NOT called.

        We patch:
          - DATA_PATH, DOWNLOAD_PATH, KAGGLE_DATA_URL to override the constants with test values.
          - os.path.exists to control whether the file is seen as existing.

        The test sets os.path.exists to return True, so check_and_download_data() should detect
        that the file exists and therefore not call download_dataset.
        """
        mock_exists.return_value = True
        with patch("heart_disease_prediction.data.data_loader.download_dataset") as mock_download:
            check_and_download_data()
            mock_download.assert_not_called()

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.DOWNLOAD_PATH", TEST_DOWNLOAD_PATH)
    @patch("heart_disease_prediction.data.data_loader.KAGGLE_DATA_URL", TEST_KAGGLE_URL)
    @patch("os.path.exists")
    def test_check_and_download_data_not_exists(self, mock_exists):
        """
        Test that if the dataset does NOT exist, the download_dataset function IS called.

        Here, os.path.exists is patched to return False, simulating that the file is missing.
        As a result, check_and_download_data() should call download_dataset.
        """
        mock_exists.return_value = False
        with patch("heart_disease_prediction.data.data_loader.download_dataset") as mock_download:
            check_and_download_data()
            mock_download.assert_called_once()

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.DOWNLOAD_PATH", TEST_DOWNLOAD_PATH)
    @patch("heart_disease_prediction.data.data_loader.KAGGLE_DATA_URL", TEST_KAGGLE_URL)
    @patch("heart_disease_prediction.data.data_loader.requests.get")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_download_dataset_success(
            self,
            mock_exists,
            mock_makedirs,
            mock_requests_get
    ):
        """
        Test successful dataset download.

        In this test, we simulate a successful HTTP response (status code 200) from the Kaggle URL.
        The patch for requests.get intercepts the network call and returns a dummy response.
        We also patch the file open function to simulate file writing without touching the disk.
        """
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test_csv_data"
        mock_requests_get.return_value = mock_response

        # Using patch on builtins.open so no actual file is created.
        with patch("builtins.open", mock_open()) as mock_file:
            download_dataset()
            # Verify that the function calls requests.get with the test URL and stream=True.
            mock_requests_get.assert_called_once_with(TEST_KAGGLE_URL, stream=True)
            # Verify that the file was opened for writing in binary mode.
            mock_file.assert_called_once_with(TEST_DATA_PATH, "wb")

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.DOWNLOAD_PATH", TEST_DOWNLOAD_PATH)
    @patch("heart_disease_prediction.data.data_loader.KAGGLE_DATA_URL", TEST_KAGGLE_URL)
    @patch("heart_disease_prediction.data.data_loader.requests.get")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_download_dataset_failure(
            self,
            mock_exists,
            mock_makedirs,
            mock_requests_get
    ):
        """
        Test failure in downloading dataset when status code is not 200.

        This test simulates a failed HTTP response (e.g., 404 status code). The test ensures that,
        in this case, no file is written to disk.
        """
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b""
        mock_requests_get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            download_dataset()
            # Verify the HTTP call was made with the test URL.
            mock_requests_get.assert_called_once_with(TEST_KAGGLE_URL, stream=True)
            # Verify that open() was not called since the download failed.
            mock_file.assert_not_called()

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.DOWNLOAD_PATH", TEST_DOWNLOAD_PATH)
    @patch("heart_disease_prediction.data.data_loader.check_and_download_data")
    def test_load_data_success(self, mock_check_and_download):
        """
        Test successful data loading.

        The test patches os.path.exists to simulate the presence of the data file and
        pandas.read_csv to return a dummy DataFrame.
        """
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv", return_value=pd.DataFrame({"col": [1, 2, 3]})) as mock_read_csv:
                data = load_data()
                # Verify that check_and_download_data was called.
                mock_check_and_download.assert_called_once()
                # Verify that pandas.read_csv was called with the correct path.
                mock_read_csv.assert_called_once_with(TEST_DATA_PATH)
                # Confirm that the returned DataFrame is not empty.
                self.assertFalse(data.empty, "DataFrame should not be empty")

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.check_and_download_data")
    def test_load_data_failure_no_file(self, mock_check_and_download):
        """
        Test load_data failure when the file is missing even after calling check_and_download_data.

        In this scenario, os.path.exists is patched to return False, causing load_data to exit.
        The test expects a SystemExit exception.
        """
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(SystemExit):
                load_data()

    @patch("heart_disease_prediction.data.data_loader.DATA_PATH", TEST_DATA_PATH)
    @patch("heart_disease_prediction.data.data_loader.check_and_download_data")
    def test_load_data_exception_during_read(self, mock_check_and_download):
        """
        Test that load_data handles exceptions during the data read operation.

        Here, we simulate an exception in pandas.read_csv (such as a parsing error) by using side_effect.
        The test ensures that if an exception occurs during reading, the function exits as expected.
        """
        with patch("os.path.exists", return_value=True):
            # Force pandas.read_csv to raise an Exception
            with patch("pandas.read_csv", side_effect=Exception("Parse error")):
                with self.assertRaises(SystemExit):
                    load_data()

if __name__ == "__main__":
    unittest.main()
