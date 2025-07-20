from argparse import ArgumentParser
import os
from typing import List, Callable

import pandas as pd
from fastvec import Word2Vec, Doc2Vec, FastvecModel, simple_preprocessing, Tokens
import time
from functools import wraps
from functools import singledispatch


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
                f.write(f"{func.__name__}: {elapsed_time:.6f}\n")
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
    df = df.sample(frac=0.1, random_state=42)
    print(f"Loaded {len(df)} food reviews from {file_path}")
    reviews = df["Text"].tolist()
    return reviews


@wall_time("walltimes/fastvec_food_reviews.txt")
def preprocess_reviews(reviews: List[str]) -> List[str]:
    """
    Preprocess food reviews by tokenizing and removing punctuation.

    Args:
        reviews (List[str]): List of food reviews.

    Returns:
        List[str]: Preprocessed food reviews.
    """
    return simple_preprocessing(reviews, deacc=True)


@singledispatch
def train_on_food_reviews(model: FastvecModel, tokens: Tokens) -> FastvecModel:
    """
    Train a FastVec model on food reviews.

    Args:
        model (FastvecModel): FastVec model to train.
        tokens (List[str]): Preprocessed food reviews.
        embedding_dim (int): Dimension of the embeddings.
        epochs (int): Number of training epochs.

    Returns:
        FastvecModel: Trained FastVec model.
    """
    raise NotImplementedError("Unsupported model type for training.")


@train_on_food_reviews.register(Doc2Vec)
@wall_time("walltimes/fastvec_food_reviews.txt")
def train_doc2vec_on_food_reviews(model: Doc2Vec, tokens: Tokens) -> Doc2Vec:
    """
    Train a Doc2Vec model on food reviews.

    Args:
        tokens (List[str]): Preprocessed food reviews.
        embedding_dim (int): Dimension of the embeddings.
        epochs (int): Number of training epochs.

    Returns:
        Doc2Vec: Trained Doc2Vec model.
    """
    return model.train(tokens)


@train_on_food_reviews.register(Word2Vec)
@wall_time("walltimes/fastvec_food_reviews.txt")
def train_word2vec_on_food_reviews(model: Word2Vec, tokens: Tokens) -> Word2Vec:
    """
    Train a Word2Vec model on food reviews.

    Args:
        tokens (List[str]): Preprocessed food reviews.
        embedding_dim (int): Dimension of the embeddings.
        epochs (int): Number of training epochs.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    return model.train(tokens)


@wall_time("walltimes/fastvec_food_reviews.txt")
def inference(model, inference_tokens):
    """
    Perform inference on the trained Doc2Vec model.

    Args:
        model (Doc2Vec): Trained Doc2Vec model.
        reviews (List[str]): List of food reviews to infer embeddings for.

    Returns:
        List[torch.Tensor]: List of embeddings for the provided reviews.
    """
    return model.get_embeddings(inference_tokens)


def to_pascal_case(input_string: str) -> str:
    """Converts a string to PascalCase (suitable for class names)."""
    words = input_string.replace(
        "_", " "
    ).split()  # Handle snake_case by replacing underscores with spaces
    pascal_cased_words = [word.capitalize() for word in words]
    return "".join(pascal_cased_words)


def main() -> None:
    parser = ArgumentParser(description="Benchmark FastVec on food reviews")
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="Dimension of the embeddings"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="food_reviews.csv",
        help="Path to the food reviews CSV file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["doc2vec", "word2vec"],
        default="doc2vec",
        help="Model to use for training",
    )
    args = parser.parse_args()

    # Load food reviews from a CSV file
    reviews = load_food_reviews(args.path)

    # seperate into train and test sets
    idx = int(len(reviews) * 0.8)
    train_reviews, test_reviews = reviews[:idx], reviews[idx:]
    print(f"Train reviews: {len(train_reviews)}, Test reviews: {len(test_reviews)}")
    tokens = preprocess_reviews(train_reviews)

    # Train Doc2Vec model on the food reviews
    model = to_pascal_case(args.model)
    model = globals()[model](embedding_dim=args.embedding_dim, epochs=10)
    model = train_on_food_reviews(
        model, tokens, embedding_dim=args.embedding_dim, epochs=10
    )

    # Example usage: Get embeddings for a specific review
    inference(model, test_reviews)

    # read wall times and generate a report
    with open("walltimes/fastvec_food_reviews.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())


if __name__ == "__main__":
    main()
