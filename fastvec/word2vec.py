from typing import List

from fastvec import Vocab
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

    # def forward(
    #     self, input_words: torch.Tensor, target_words: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Forward pass for the Word2Vec model.

    #     Args:
    #         input_words (torch.Tensor): Input word indices.
    #         target_words (torch.Tensor): Target word indices.

    #     Returns:
    #         torch.Tensor: Output embeddings.
    #     """
    #     input_embeddings = self.input_encoder(input_words)
    #     target_embeddings = self.target_encoder(target_words)

    #     # Combine the embeddings (get dot product)
    #     sim = (target_embeddings * input_embeddings).sum(dim=1)

    #     # Apply activation function
    #     output = self.activation(sim)

    #     return output

    # def build_training_set(
    #     self, documents: List[List[str]], window_size: int = 5
    # ) -> DataLoader:
    #     """
    #     Build training set from the corpus.

    #     Args:
    #         documents (List[List[str]]): List of sentences.
    #         window_size (int): Size of the context window.

    #     Returns:
    #         DataLoader: Training set as pairs of input and target words.
    #     """
    #     builder = Builder(documents, self.vocab, window_size)
    #     examples = builder.build_w2v_training()
    #     dataset = Word2VecDataset(examples)
    #     return DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # def _train(self, examples: DataLoader) -> Embedding:
    #     """Train the Word2Vec model on the provided examples.

    #     Args:
    #         examples (DataLoader): DataLoader containing training examples.

    #     Returns:
    #         Embedding: The learned embeddings.
    #     """
    #     params = list(self.input_encoder.parameters()) + list(
    #         self.target_encoder.parameters()
    #     )
    #     optimizer = torch.optim.Adam(params, lr=0.001)
    #     for _ in range(self.epochs):
    #         for batch in examples:
    #             input_words = batch["input_words"].unsqueeze(1).to(self.device)
    #             target_words = batch["target_words"].unsqueeze(1).to(self.device)
    #             labels = batch["label"].to(self.device)

    #             # Forward pass
    #             outputs = self.forward(input_words, target_words)

    #             # Compute loss
    #             loss = nn.BCELoss()(outputs, labels)

    #             # Backward pass and optimization
    #             self.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     return self.save_embeddings(examples)

    # def save_embeddings(self, examples: DataLoader) -> Embedding:
    #     """
    #     Save the learned embeddings.

    #     Returns:
    #         Embedding: The learned embeddings.
    #     """
    #     embeddings = Embedding(self.embedding_dim)
    #     for batch in examples:
    #         input_words = batch["input_words"].unsqueeze(1)
    #         vectors = self.input_encoder(input_words)
    #         embeddings.add_vectors(
    #             input_words.flatten().to(torch.int64).tolist(), vectors.tolist()
    #         )
    #     return embeddings

    # def train(self, tokens: Tokens, window_size: int = 5) -> None:
    #     """
    #     Train the Word2Vec model on the given corpus.

    #     Args:
    #         corpus (List[str]): List of sentences.
    #         window_size (int): Size of the context window.

    #     Returns:
    #         Embedding: The learned embeddings.
    #     """
    #     self.build_vocab(tokens.flatten())
    #     examples = self.build_training_set(tokens.tokens, window_size)
    #     self.embeddings = self._train(examples)

    # def get_embeddings(self, words: List[str]) -> List[List[float]]:
    #     """
    #     Get the learned embeddings.

    #     Returns:
    #         torch.Tensor: The learned embeddings.

    #     Raises:
    #         ValueError: If the model has not been trained yet.
    #     """
    #     if hasattr(self, "embeddings"):
    #         indices = self.vocab.get_ids(words)
    #         return self.embeddings.get_vectors(indices)
    #     else:
    #         raise ValueError(
    #             "Model has not been trained yet. Call 'train' method first."
    #         )

    # def __getitem__(self, word: str) -> List[float]:
    #     """
    #     Get the embedding for a specific word.

    #     Args:
    #         word (str): The word to get the embedding for.

    #     Returns:
    #         List[float]: The embedding vector for the word.
    #     """
    #     return self.get_embeddings([word])[0]
