from abc import ABC, abstractmethod


class FastvecModel(ABC):
    """
    Abstract base class for models in FastVec.
    """

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model on the provided data.
        """
        pass

    @abstractmethod
    def get_embeddings(self, *args, **kwargs):
        """
        Get the learned embeddings from the model.
        """
        pass

    @abstractmethod
    def build_vocab(self, *args, **kwargs):
        """
        Build the vocabulary from the provided data.
        """
        pass

    @abstractmethod
    def build_training_set(self, *args, **kwargs):
        """
        Build the training set from the provided data.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the model from the specified path.

        Args:
            path (str): Path to load the model from.
        """
        pass
