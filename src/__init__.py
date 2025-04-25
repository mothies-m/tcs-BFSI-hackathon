from .dataset import download_dataset
from .preprocess import preprocess_dataset
from .modelling import build_model


def run_pipeline():
    dataset_path = download_dataset(
        "kabure/german-credit-data-with-risk",
    )
    x, y = preprocess_dataset(dataset_path)
    build_model(x, y)


if __name__ == "__main__":
    run_pipeline()
