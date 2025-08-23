# from argparse import ArgumentParser
import pyperf

import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_food_reviews, train_test_split


# def parse_args():
#     parser = ArgumentParser(description="Benchmark Gensim on food reviews")
#     parser.add_argument(
#         "--embedding_dim", type=int, default=64, help="Dimension of the embeddings"
#     )
#     parser.add_argument(
#         "-p",
#         "--path",
#         type=str,
#         default="food_reviews.csv",
#         help="Path to the food reviews CSV file",
#     )
#     parser.add_argument(
#         "-m",
#         "--model",
#         type=str,
#         choices=["doc2vec", "word2vec"],
#         default="doc2vec",
#         help="Model to use for training",
#     )
#     return parser.parse_args()


def train(model, examples):
    """
    Train a Gensim model on food reviews.

    Args:
        model (Doc2Vec or Word2Vec): Gensim model to train.
        examples (List[List[str]]): Preprocessed food reviews.

    Returns:
        Doc2Vec or Word2Vec: Trained Gensim model.
    """
    if isinstance(model, Doc2Vec):
        model.build_vocab(examples)
        model.train(examples, total_examples=model.corpus_count, epochs=model.epochs)
    else:
        model.build_vocab(examples)
        model.train(examples, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def inference(model, inference_tokens):
    """
    Perform inference on the trained Gensim model.

    Args:
        model (Doc2Vec or Word2Vec): Trained Gensim model.
        inference_tokens (List[List[str]]): Tokens to infer embeddings for.

    Returns:
        List[torch.Tensor]: List of embeddings for the provided tokens.
    """
    if isinstance(model, Doc2Vec):
        return [model.infer_vector(token) for token in inference_tokens]
    return [
        model.wv[token]
        for tokens in inference_tokens
        for token in tokens
        if token in model.wv
    ]


def main():
    path = "amazon-reviews/Reviews.csv"
    model = "word2vec"
    embedding_dim = 64
    runner = pyperf.Runner()
    # Load food reviews from a CSV file
    reviews = load_food_reviews(path)
    # Separate into train and test sets
    train_reviews, test_reviews = train_test_split(reviews)
    if model == "doc2vec":
        model = Doc2Vec(
            vector_size=embedding_dim,
            min_count=5,
            hs=0,
            epochs=5,
            negative=5,
        )
    else:
        model = Word2Vec(
            vector_size=embedding_dim,
            min_count=5,
            hs=0,
            epochs=5,
            negative=5,
            sg=0,
        )
    tokens = [
        gensim.utils.simple_preprocess(review, deacc=True) for review in train_reviews
    ]

    examples = (
        [TaggedDocument(tokens, [i]) for i, tokens in enumerate(tokens)]
        if model == "doc2vec"
        else tokens
    )

    runner.bench_func("train_gensim", train, model, examples)
    inference_tokens = [
        gensim.utils.simple_preprocess(review, deacc=True) for review in test_reviews
    ]

    print("Number of Gensim tokens:", len(model.wv))
    runner.bench_func("inference_gensim", inference, model, inference_tokens)


if __name__ == "__main__":
    main()
