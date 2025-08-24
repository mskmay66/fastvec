import pyperf
from fastvec import (
    FastvecModel,
    Word2Vec,
    Doc2Vec,
    simple_preprocessing,
    Tokens,
    Dataset,
    train_word2vec,
)
from utils import load_food_reviews, train_test_split


# def parse_args():
#     parser = ArgumentParser(description="Benchmark FastVec on food reviews")
#     parser.add_argument(
#         "--embedding_dim", type=int, default=64, help="Dimension of the embeddings")
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


def train(model: FastvecModel, examples: Dataset) -> FastvecModel:
    """
    Train a FastVec model on food reviews.

    Args:
        model (FastvecModel): FastVec model to train.
        examples (Dataset): Training set for the model.

    Returns:
        FastvecModel: Trained FastVec model.
    """
    model.embeddings = train_word2vec(
        examples, model.embedding_dim, 128, 3, model.lr, model.epochs
    )


def inference(model: FastvecModel, inference_tokens: Tokens):
    """
    Perform inference on the trained FastVec model.

    Args:
        model (FastvecModel): Trained FastVec model.
        inference_tokens (Tokens): Tokens to infer embeddings for.

    Returns:
        List[torch.Tensor]: List of embeddings for the provided tokens.
    """
    return model.get_embeddings(inference_tokens)


def main():
    path = "amazon-reviews/Reviews.csv"
    model = "word2vec"
    embedding_dim = 64
    # Load food reviews from a CSV file
    reviews = load_food_reviews(path)
    train_reviews, test_reviews = train_test_split(reviews)

    tokens = simple_preprocessing(
        train_reviews, deacc=True, lowercase=True, split_pattern=r"\W+"
    )
    print(len(tokens.flatten()), "tokens in training set")

    runner = pyperf.Runner()
    # Initialize FastVec model
    model = (
        Word2Vec(embedding_dim, epochs=5)
        if model == "word2vec"
        else Doc2Vec(embedding_dim, epochs=5)
    )
    model.build_vocab(tokens.flatten())
    examples = model.build_training_set(tokens.tokens, window_size=5)

    print("Vocab length:", len(model.vocab.valid_ids))

    runner.bench_func("train_fastvec", train, model, examples)
    model.train(tokens)

    runner.bench_func(
        "inference_fastvec",
        inference,
        model,
        simple_preprocessing(test_reviews, deacc=True).flatten(),
    )


if __name__ == "__main__":
    main()
