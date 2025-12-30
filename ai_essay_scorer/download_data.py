import fire
import kagglehub
from kagglehub import KaggleDatasetAdapter


def download_dataset(kaggle_dataset: str, dataset_file: str, path_to_save: str) -> None:
    """Download a dataset from Kaggle and save it as CSV.

    Fetches a dataset from Kaggle using the kagglehub library, converts it to
    a pandas DataFrame, and saves it as a CSV file to the specified location.
    Also prints a preview of the first 3 rows for verification.

    Args:
        kaggle_dataset (str): Kaggle dataset identifier in format 'owner/dataset-name'.
            Example: 'datasets/cornell-university/arxiv'
        dataset_file (str): Path or identifier of the specific file within the dataset
            to load. Example: 'arxiv.csv' or 'data/train.parquet'
        path_to_save (str): Local file path where the CSV will be saved.
            Example: './data/arxiv_data.csv'

    Returns:
        None

    Example:
        >>> download_dataset(
        ...     kaggle_dataset="datasets/cornell-university/arxiv",
        ...     dataset_file="arxiv.csv",
        ...     path_to_save="./data/arxiv.csv",
        ... )
        Dataset downloaded and save in ./data/arxiv.csv
        Dataset rows example:
           ...

    Note:
        - Requires Kaggle API credentials configured (~/.kaggle/kaggle.json)
        - The index column is not saved in the CSV (index=False)
    """
    df = kagglehub.dataset_load(
        handle=kaggle_dataset, adapter=KaggleDatasetAdapter.PANDAS, path=dataset_file
    )
    df = df[["full_text", "score"]].rename(columns={"full_text": "text", "score": "labels"})
    df.to_csv(path_to_save, index=False)
    print(f"Dataset downloaded and save in {path_to_save}")
    print("Dataset rows example:")
    print(df.head(3))


if __name__ == "__main__":
    fire.Fire(download_dataset)
