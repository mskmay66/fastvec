from __future__ import annotations
from typing import List
import pickle

from fastvec import Vocab, Builder, Tokens, train_word2vec
from .model import FastvecModel


class Word2Vec(FastvecModel):
    def __init__(self, embedding_dim, epochs=100):
        super(Word2Vec, self).__init__()
        self.embedding_dim = embedding_dim
        self.epochs = epochs

        self.vocab = None
        self.embeddings = None

    def build_vocab(self, corpus: List[str]) -> None:
        """Build vocabulary from the corpus.

        Args:
            corpus (List[str]): List of words.
        """
        self.vocab = Vocab.from_words(corpus)

    def build_training_set(self, documents: List[List[str]], window_size: int = 5):
        """
        Build the training set from the provided data.
        """
        builder = Builder(documents, self.vocab, window_size)
        return builder.build_training()

    def train(self, tokens: Tokens, window_size: int = 5) -> None:
        """
        Train the Word2Vec model on the given corpus.

        Args:
            tokens (Tokens): List of words.
            window_size (int): Size of the context window.
        """
        self.build_vocab(tokens.flatten())
        examples = self.build_training_set(tokens.tokens, window_size)
        self.embeddings = train_word2vec(
            examples,
            embedding_dim=self.embedding_dim,
            epochs=self.epochs,
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
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Word2Vec:
        with open(path, "rb") as f:
            return pickle.load(f)
