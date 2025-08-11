from functools import wraps
from typing import Callable, List, Tuple
import os
import time
import pandas as pd


def wall_time(file_path: str) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "a") as f:
                f.write(f"{func.__name__}: {elapsed_time:.6f} Seconds\n")
            return result

        return wrapper

    return decorator


def load_food_reviews(file_path: str) -> List[str]:
    """
    Load food reviews from a CSV file and preprocess the text.

    Args:
        file_path (str): Path to the CSV file containing food reviews.

    Returns:
        List[str]: Preprocessed food reviews.
    """
    df = pd.read_csv(file_path)
    df = df.sample(frac=0.001, random_state=42)
    print(f"Loaded {len(df)} food reviews from {file_path}")
    reviews = df["Text"].tolist()
    return reviews


def to_pascal_case(input_string: str) -> str:
    """Converts a string to PascalCase (suitable for class names)."""
    words = input_string.split("2")
    pascal_cased_words = [word.title() for word in words]
    return "2".join(pascal_cased_words)


def load_wall_times(path: str) -> None:
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())


def train_test_split(reviews: List[str]) -> Tuple[List[str], List[str]]:
    idx = int(len(reviews) * 0.8)
    train_reviews, test_reviews = reviews[:idx], reviews[idx:]
    return train_reviews, test_reviews
