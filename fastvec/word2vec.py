import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List

from fastvec import Vocab, Embedding, Builder


class Word2VecDataset(Dataset):

    def __init__(self, examples):
        super(Word2VecDataset, self).__init__()
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_words, target_words, label = self.examples[idx]
        return {
            'input_words': torch.tensor(input_words, dtype=torch.float32),
            'target_words': torch.tensor(target_words, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


class Word2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, epochs=100):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.epochs = epochs    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_encoder = nn.Linear(2, embedding_dim)
        self.target_encoder = nn.Linear(2, embedding_dim)
        self.activation = nn.Sigmoid()

        self.input_encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.activation.to(self.device)
        
        self.vocab = None
        self.embeddings = None

    
    def build_vocab(self, corpus: List[str]) -> None:
        """ Build vocabulary from the corpus.

        Args:
            corpus (List[str]): List of sentences.
        """
        self.vocab = Vocab.from_words(corpus)


    def forward(self, input_words: torch.Tensor, target_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Word2Vec model.
        
        Args:
            input_words (torch.Tensor): Input word indices.
            target_words (torch.Tensor): Target word indices.
        
        Returns:
            torch.Tensor: Output embeddings.
        """
        input_embeddings = self.input_encoder(input_words.float())
        target_embeddings = self.target_encoder(target_words.float())
        
        # Combine the embeddings
        sim = torch.dot(input_embeddings, target_embeddings)
        
        # Apply activation function
        output = self.activation(sim)
        
        return output


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
        examples = builder.build_w2v_training()
        datset = Word2VecDataset(examples)
        return DataLoader(datset, batch_size=32, shuffle=True)

    def _train(self, examples: DataLoader) -> Embedding:
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
        return self.save_embeddings(examples)
        
    
    def save_embeddings(self, examples) -> Embedding:
        """
        Save the learned embeddings.
        
        Returns:
            Embedding: The learned embeddings.
        """
        embeddings = Embedding(self.embedding_dim)
        for batch in examples:
            input_words = batch['input_words']
            vectors = self.input_encoder(input_words)
            embeddings.add_vectors(input_words, vectors)
        return embeddings


    def train(self, corpus: List[str], window_size: int = 5) -> None:
        """
        Train the Word2Vec model on the given corpus.
        
        Args:
            corpus (List[str]): List of sentences.
            window_size (int): Size of the context window.
        
        Returns:
            Embedding: The learned embeddings.
        """
        self.build_vocab(corpus)
        examples = self.build_training_set(corpus, window_size)
        self.embeddings = self._train(examples)
    

    def get_embeddings(self, words: List[str]) -> List[List[float]]:
        """
        Get the learned embeddings.
        
        Returns:
            torch.Tensor: The learned embeddings.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if hasattr(self, 'embeddings'):
            indices = self.vocab.get_ids(words)
            return self.embeddings.get_vectors(indices)
        else:
            raise ValueError("Model has not been trained yet. Call 'train' method first.")
    
    def __getitem__(self, word: str) -> List[float]:
        """
        Get the embedding for a specific word.
        
        Args:
            word (str): The word to get the embedding for.
        
        Returns:
            List[float]: The embedding vector for the word.
        """
        return self.get_embeddings([word])[0]



