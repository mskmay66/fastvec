import torch
from torch import nn
from typing import List
from torch.utils.data import Dataset, DataLoader

from .word2vec import Word2Vec
from fastvec import Vocab, Embedding, Builder


class Doc2VecDataset(Dataset):

    def __init__(self, examples):
        super(Doc2VecDataset, self).__init__()
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        doc_id, input_words, target_words, label = self.examples[idx]
        return {
            'doc_id': torch.tensor(doc_id, dtype=torch.float32),
            'input_words': torch.tensor(input_words, dtype=torch.float32),
            'target_words': torch.tensor(target_words, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class Doc2Vec(Word2Vec):
    """
    Doc2Vec model for document embedding.
    
    Inherits from Word2Vec and extends its functionality to handle documents.
    """

    def __init__(self, vocab_size, embedding_dim, epochs=100):
        super(Doc2Vec, self).__init__(vocab_size, embedding_dim, epochs)
        self.document_encoder = nn.Linear(2, embedding_dim)
        self.document_encoder.to(self.device)

    def build_training_set(self, corpus: List[str], window_size: int = 5) -> List[tuple]:
        """
        Build training set from the corpus.
        
        Args:
            corpus (List[str]): List of sentences.
            window_size (int): Size of the context window.
        
        Returns:
            List[tuple]: Training set as pairs of input and target words.
        """
        builder = Builder(corpus, self.vocab, window_size)
        examples = builder.build_d2v_training()
        datset = Doc2VecDataset(examples)
        return DataLoader(datset, batch_size=32, shuffle=True)
    
    def forward(self, input_words: torch.Tensor, target_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Word2Vec model.
        
        Args:
            input_words (torch.Tensor): Input word indices.
            target_words (torch.Tensor): Target word indices.
        
        Returns:
            torch.Tensor: Output embeddings.
        """
        doc_embeddings = self.document_encoder(input_words.float())
        input_embeddings = self.input_encoder(input_words.float())
        target_embeddings = self.target_encoder(target_words.float())
        
        # Combine the embeddings
        sim = torch.dot((doc_embeddings + input_embeddings) / 2, target_embeddings)
        
        # Apply activation function
        output = self.activation(sim)
        
        return output
    
    def _train(self, examples: DataLoader) -> None:
        """ Train the Word2Vec model on the provided examples.

        Args:
            examples (DataLoader): DataLoader containing training examples.

        Returns:
            Embedding: The learned embeddings.
        """
        params = list(self.input_encoder.parameters()) + list(self.target_encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
        for _ in range(self.epochs):
            for batch in examples:
                doc_id = batch['doc_id']
                input_words = batch['input_words']
                target_words = batch['target_words']
                labels = batch['label']

                # Forward pass
                outputs = self.forward(input_words, target_words)

                # Compute loss
                loss = nn.BCELoss()(outputs, labels)

                # Backward pass and optimization
                self.zero_grad()
                loss.backward()
                optimizer.step()
        

    # def get_embeddings(self, words: List[str]) -> List[List[float]]:
    #     """
    #     Get the learned embeddings.
        
    #     Returns:
    #         torch.Tensor: The learned embeddings.

    #     Raises:
    #         ValueError: If the model has not been trained yet.
    #     """
        