from __future__ import annotations
from typing import List
import pickle

from fastvec import Vocab, Builder, Tokens, Dataset, train_word2vec, Embedding
from .model import FastvecModel


class Word2Vec(FastvecModel):
    """Word2Vec model for learning word embeddings.
    This model inherits from FastvecModel and implements the Word2Vec algorithm
    for learning word embeddings from a corpus of text.
    It builds a vocabulary from the corpus, creates training examples,
    and trains the model to learn word embeddings.
    It provides methods to build the vocabulary, train the model,
    get embeddings for specific words, and save/load the model.
    It uses the Builder class to create training examples and the
    train_word2vec function to train the model.

    Attributes:
        embedding_dim (int): Dimension of the word embeddings.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for training.
        vocab (Vocab): Vocabulary built from the corpus.
        embeddings (Tokens): Learned word embeddings.

    Methods:
        build_vocab(corpus: List[str]) -> None:
            Build vocabulary from the corpus.
        build_training_set(documents: List[List[str]], window_size: int = 5):
            Build the training set from the provided data.
        train(tokens: Tokens, window_size: int = 5) -> None:
            Train the Word2Vec model on the given corpus.
        get_embeddings(words: List[str]) -> List[List[float]]:
            Get the learned embeddings for specific words.
        save(path: str) -> None:
            Save the Word2Vec model to the specified path.
        load(path: str) -> Word2Vec:
            Load a Word2Vec model from the specified path.
    """

    def __init__(
        self, embedding_dim, epochs=5, lr=0.01, batch_size=128, min_count=5
    ) -> None:
        super(Word2Vec, self).__init__()
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.min_count = min_count

        self.vocab = None
        self.embeddings = None

    def build_vocab(self, corpus: List[str]) -> None:
        """Build vocabulary from the corpus.

        Args:
            corpus (List[str]): List of words.
        """
        self.vocab = Vocab.from_words(corpus, self.min_count)

    def build_training_set(
        self, documents: List[List[str]], window_size: int = 5, num_neg_samples: int = 5
    ) -> Dataset:
        """
        Build the training set from the provided data.

        Args:
            documents (List[List[str]]): List of documents, where each document is a list of words.
            window_size (int): Size of the context window.

        Returns:
            Dataset: The training set containing word pairs and their contexts.
        """
        builder = Builder(documents, self.vocab, window_size)
        return builder.build_training(
            num_neg=num_neg_samples, batch_size=self.batch_size
        )

    def train(self, tokens: Tokens, window_size: int = 5) -> None:
        """
        Train the Word2Vec model on the given corpus.

        Args:
            tokens (Tokens): List of words.
            window_size (int): Size of the context window.
        """
        self.embeddings = Embedding(self.embedding_dim)
        self.build_vocab(tokens.flatten())
        examples = self.build_training_set(tokens.tokens, window_size)
        self.embeddings = train_word2vec(
            examples,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
        )

    def get_embeddings(self, words: List[str]) -> List[List[float]]:
        """
        Get the learned embeddings.

        Args:
            words (List[str]): List of words to get embeddings for.

        Returns:
            List[List[float]]: The learned embeddings for the provided words.
        """
        if self.embeddings is not None:
            indices = self.vocab.get_ids(words)
            return self.embeddings.get_vectors(indices)
        else:
            raise ValueError(
                "Model has not been trained yet. Call 'train' method first."
            )

    def save(self, path: str) -> None:
        """Saves the Word2Vec model to the specified path.

        Args:
            path (str): The path where the model will be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Word2Vec:
        """Reads a Word2Vec model from the specified path.

        Args:
            path (str): The location of the model.

        Returns:
            Word2Vec: The loaded Word2Vec model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
