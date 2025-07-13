from argparse import ArgumentParser

import pandas as pd
from fastvec import Word2Vec, Doc2Vec, simple_preprocessing
import time
from functools import wraps


def wall_time(file_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            with open(file_path, "a") as f:
                f.write(f"{func.__name__}:{elapsed_time:.2f}")
            return result

        return wrapper

    return decorator


def load_food_reviews(file_path):
    """
    Load food reviews from a CSV file and preprocess the text.

    Args:
        file_path (str): Path to the CSV file containing food reviews.

    Returns:
        List[str]: Preprocessed food reviews.
    """
    df = pd.read_csv(file_path)
    reviews = df["Text"].tolist()
    return reviews.sample(frac=1, random_state=42)


@wall_time("benchmarking/fastvec_food_reviews.txt")
def preprocess_reviews(reviews):
    """
    Preprocess food reviews by tokenizing and removing punctuation.

    Args:
        reviews (List[str]): List of food reviews.

    Returns:
        List[str]: Preprocessed food reviews.
    """
    return simple_preprocessing(reviews, deacc=True)


@wall_time("benchmarking/fastvec_food_reviews.txt")
def train_doc2vec_on_food_reviews(tokens, embedding_dim=64, epochs=10):
    """
    Train a Doc2Vec model on food reviews.

    Args:
        tokens (List[str]): Preprocessed food reviews.
        embedding_dim (int): Dimension of the embeddings.
        epochs (int): Number of training epochs.

    Returns:
        Doc2Vec: Trained Doc2Vec model.
    """
    model = Doc2Vec(embedding_dim=embedding_dim, epochs=epochs)
    model.train(tokens)
    return model


@wall_time("benchmarking/fastvec_food_reviews.txt")
def train_word2vec_on_food_reviews(tokens, embedding_dim=64, epochs=10):
    """
    Train a Word2Vec model on food reviews.

    Args:
        tokens (List[str]): Preprocessed food reviews.
        embedding_dim (int): Dimension of the embeddings.
        epochs (int): Number of training epochs.

    Returns:
        Word2Vec: Trained Word2Vec model.
    """
    model = Word2Vec(embedding_dim=embedding_dim, epochs=epochs)
    model.train(tokens)
    return model


@wall_time("benchmarking/fastvec_food_reviews.txt")
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


def main():
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

    tokens = preprocess_reviews(train_reviews)

    # Train Doc2Vec model on the food reviews
    if args.model == "word2vec":
        model = train_word2vec_on_food_reviews(
            tokens, embedding_dim=args.embedding_dim, epochs=10
        )
    else:
        model = train_doc2vec_on_food_reviews(tokens, embedding_dim=64, epochs=10)

    # Example usage: Get embeddings for a specific review
    inference_tokens = preprocess_reviews(test_reviews)
    inference(model, inference_tokens)

    # read wall times and generate a report
    with open("benchmarking/fastvec_food_reviews.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())


if __name__ == "__main__":
    main()
