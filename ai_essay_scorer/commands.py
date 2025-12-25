import fire
from download_data import download_dataset

if __name__ == "__main__":
    fire.Fire({"download_data": download_dataset})
