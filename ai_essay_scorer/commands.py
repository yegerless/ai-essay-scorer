import fire
from download_data import download_dataset
from train import train

if __name__ == "__main__":
    fire.Fire({"download_data": download_dataset, "train": train})
