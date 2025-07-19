from fastvec import Word2Vec, Word2VecDataset, Builder
import torch
import pytest


@pytest.fixture(scope="module")
def w2v_model():
    w2v = Word2Vec(embedding_dim=64, epochs=10)
    return w2v


def test_w2v_creation(w2v_model):
    assert w2v_model.embedding_dim == 64
    assert w2v_model.epochs == 10
    assert w2v_model.vocab is None
    assert w2v_model.embeddings is None
    assert w2v_model.input_encoder is not None
    assert w2v_model.target_encoder is not None
    assert w2v_model.activation is not None
    assert w2v_model.input_encoder.weight.shape == (64, 1)
    assert w2v_model.target_encoder.weight.shape == (64, 1)


def test_w2v_forward(w2v_model):
    input_words = torch.tensor([[1], [2]], dtype=torch.float32)
    target_words = torch.tensor([[3], [4]], dtype=torch.float32)

    output = w2v_model.forward(input_words, target_words)

    assert output.shape == (2, 64)  # Output should match embedding dimension
    assert output.dtype == torch.float32  # Output should be of float type


def test_w2v_build_vocab(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    w2v_model.build_vocab(corpus)

    assert w2v_model.vocab is not None
    assert len(w2v_model.vocab) > 0  # Vocabulary should not be empty
    assert "hello" in w2v_model.vocab.word_to_index
    assert "world" in w2v_model.vocab.word_to_index
    assert "fastvec" in w2v_model.vocab.word_to_index
    assert "is" in w2v_model.vocab.word_to_index
    assert "great" in w2v_model.vocab.word_to_index


def test_w2v_dataset(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    w2v_model.build_vocab(corpus)

    builder = Builder(corpus, w2v_model.vocab, window_size=1)
    examples = builder.build_w2v_training()

    dataset = Word2VecDataset(examples)

    assert len(dataset) == len(examples)

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "input_words" in sample
    assert "target_words" in sample
    assert "label" in sample
    assert sample["input_words"].dtype == torch.float32
    assert sample["target_words"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32


def test_w2v_training_set(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    w2v_model.build_vocab(corpus)

    _ = Builder(corpus, w2v_model.vocab, window_size=1)
    training_set = w2v_model.build_training_set(corpus, window_size=1)

    assert isinstance(training_set, torch.utils.data.DataLoader)
    assert len(training_set.dataset) > 0  # Training set should not be empty
    for batch in training_set:
        assert "input_words" in batch
        assert "target_words" in batch
        assert "label" in batch
        assert batch["input_words"].dtype == torch.float32
        assert batch["target_words"].dtype == torch.float32
        assert batch["label"].dtype == torch.float32


def test_w2v_training(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    w2v_model.build_vocab(corpus)

    training_set = w2v_model.build_training_set(corpus, window_size=1)

    # Simulate training
    for epoch in range(1):
        for batch in training_set:
            input_words = batch["input_words"]
            target_words = batch["target_words"]
            output = w2v_model.forward(input_words, target_words)
            assert output.shape == (input_words.shape[0], w2v_model.embedding_dim)

    assert (
        w2v_model.embeddings is not None
    )  # Ensure embeddings are created after training


def test_w2v_embedding(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    w2v_model.build_vocab(corpus)

    training_set = w2v_model.build_training_set(corpus, window_size=1)

    # Simulate training
    for _ in range(1):
        for batch in training_set:
            input_words = batch["input_words"]
            target_words = batch["target_words"]
            _ = w2v_model.forward(input_words, target_words)

    embeddings = w2v_model.get_embeddings()

    assert embeddings is not None
    assert embeddings.shape[0] == len(w2v_model.vocab)  # Number of words in vocab
    assert embeddings.shape[1] == w2v_model.embedding_dim  # Embedding dimension
