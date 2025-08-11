from argparse import ArgumentParser

import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import wall_time, load_food_reviews, train_test_split, load_wall_times


@wall_time("walltimes/gensim_food_reviews.txt")
def preprocess_reviews(reviews, doc2vec=False):
    if doc2vec:
        return [
            TaggedDocument(gensim.utils.simple_preprocess(review, deacc=True), [i])
            for i, review in enumerate(reviews)
        ]
    return [gensim.utils.simple_preprocess(review, deacc=True) for review in reviews]


@wall_time("walltimes/gensim_food_reviews.txt")
def build_training_set(model, tokens):
    """
    Build the training set for the Gensim model.

    Args:
        model (Doc2Vec or Word2Vec): Gensim model to build the training set for.
        tokens (List[List[str]]): Preprocessed food reviews.
        window_size (int): Size of the context window.

    Returns:
        None: The model's vocabulary is built in place.
    """
    model.build_vocab(tokens)


@wall_time("walltimes/gensim_food_reviews.txt")
def train_on_food_reviews(model, tokens):
    """
    Train a Gensim model on food reviews.

    Args:
        model (Doc2Vec or Word2Vec): Gensim model to train.
        tokens (List[List[str]]): Preprocessed food reviews.

    Returns:
        Doc2Vec or Word2Vec: Trained Gensim model.
    """
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)


@wall_time("walltimes/gensim_food_reviews.txt")
def inference(model, inference_tokens):
    if isinstance(model, Doc2Vec):
        # For Doc2Vec, we need to infer vectors for each document
        return [model.infer_vector(token) for token in inference_tokens]
    return [
        model.wv[token]
        for tokens in inference_tokens
        for token in tokens
        if token in model.wv
    ]


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
    train_reviews, test_reviews = train_test_split(reviews)
    print(f"Train reviews: {len(train_reviews)}, Test reviews: {len(test_reviews)}")

    if args.model == "doc2vec":
        model = Doc2Vec(
            vector_size=args.embedding_dim,
            min_count=1,
            hs=0,
            epochs=10,
            negative=5,
        )
        doc2vec = True
    elif args.model == "word2vec":
        model = Word2Vec(
            vector_size=args.embedding_dim,
            min_count=5,
            hs=0,
            epochs=10,
            negative=5,
            sg=0,
        )
        doc2vec = False

    tokens = preprocess_reviews(train_reviews, doc2vec=doc2vec)
    build_training_set(model, tokens)
    train_on_food_reviews(model, tokens)
    print(f"Trained {args.model} model with {args.embedding_dim} dimensions.")
    inference_tokens = preprocess_reviews(test_reviews)

    inference(model, inference_tokens)

    load_wall_times("walltimes/gensim_food_reviews.txt")


if __name__ == "__main__":
    main()
