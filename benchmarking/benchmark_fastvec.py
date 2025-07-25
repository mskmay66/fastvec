from argparse import ArgumentParser
from typing import List

from fastvec import FastvecModel, simple_preprocessing, Tokens
from utils import (
    wall_time,
    load_food_reviews,
    to_pascal_case,
    load_wall_times,
    train_test_split,
)


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


@wall_time("walltimes/fastvec_food_reviews.txt")
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
    model.train(tokens)


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
    train_reviews, test_reviews = train_test_split(reviews)
    print(f"Train reviews: {len(train_reviews)}, Test reviews: {len(test_reviews)}")
    tokens = preprocess_reviews(train_reviews)

    # Train Doc2Vec model on the food reviews
    model = to_pascal_case(args.model)
    model = globals()[model](embedding_dim=args.embedding_dim, epochs=10)
    train_on_food_reviews(model, tokens)

    # Example usage: Get embeddings for a specific review
    inference(model, test_reviews)

    # read wall times and generate a report
    load_wall_times("walltimes/fastvec_food_reviews.txt")


if __name__ == "__main__":
    main()
