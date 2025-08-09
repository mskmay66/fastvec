from typing import List

from fastvec import Word2Vec, simple_preprocessing, infer_doc_vectors


class Doc2Vec(Word2Vec):
    """
    Doc2Vec model for document embedding.

    Inherits from Word2Vec and extends its functionality to handle documents.
    """

    def __init__(
        self,
        embedding_dim,
        epochs=100,
        lr=0.01,
        batch_size=32,
        inference_epochs=10,
        min_count=5,
    ):
        super(Doc2Vec, self).__init__(embedding_dim, epochs, lr, batch_size, min_count)
        self.inference_epochs = inference_epochs

    def get_embeddings(self, documents: List[str]) -> List[List[float]]:
        """
        Get the learned embeddings for a list of documents.

        Args:
            documents (List[str]): List of documents to infer embeddings for.

        Returns:
            List[List[float]]: The learned embeddings for the provided documents.
        """
        tokens = simple_preprocessing(documents, deacc=True)
        word_vecs = super().get_embeddings(tokens.flatten())
        return infer_doc_vectors(word_vecs, epochs=self.inference_epochs, lr=self.lr)
