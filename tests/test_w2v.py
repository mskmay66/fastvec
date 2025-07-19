from fastvec import Word2Vec, Word2VecDataset, Builder, simple_preprocessing
import torch
import pytest


@pytest.fixture(scope="module")
def w2v_model():
    w2v = Word2Vec(embedding_dim=64, epochs=10)
    return w2v


@pytest.fixture(scope="module")
def trained_w2v_model(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    tokens = simple_preprocessing(corpus, deacc=True)
    w2v_model.train(tokens, window_size=2)
    return w2v_model


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

    assert output.shape == (2,)  # Output should match embedding dimension
    assert output.dtype == torch.float32  # Output should be of float type


def test_w2v_build_vocab(w2v_model):
    corpus = ["hello", "world", "fastvec", "is", "great"]
    w2v_model.build_vocab(corpus)

    assert w2v_model.vocab is not None
    assert w2v_model.vocab.size > 0  # Vocabulary should not be empty
    assert "hello" in w2v_model.vocab.word_to_id
    assert "world" in w2v_model.vocab.word_to_id
    assert "fastvec" in w2v_model.vocab.word_to_id
    assert "is" in w2v_model.vocab.word_to_id
    assert "great" in w2v_model.vocab.word_to_id


def test_w2v_dataset(w2v_model):
    corpus = ["hello world", "fastvec is great"]
    tokens = simple_preprocessing(corpus, deacc=True)
    w2v_model.build_vocab(tokens.flatten())
    builder = Builder(tokens.tokens, w2v_model.vocab, 3)
    examples = builder.build_w2v_training()

    dataset = Word2VecDataset(examples)
    assert len(dataset) == len(examples)
    assert len(examples) > 0  # Ensure examples are created

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
    tokens = simple_preprocessing(corpus, deacc=True)
    w2v_model.build_vocab(tokens.flatten())
    training_set = w2v_model.build_training_set(tokens.tokens, window_size=3)

    assert isinstance(training_set, torch.utils.data.DataLoader)
    assert len(training_set.dataset) > 0  # Training set should not be empty
    for batch in training_set:
        assert "input_words" in batch
        assert "target_words" in batch
        assert "label" in batch
        assert batch["input_words"].dtype == torch.float32
        assert batch["target_words"].dtype == torch.float32
        assert batch["label"].dtype == torch.float32


def test_w2v_training(trained_w2v_model):
    assert trained_w2v_model.vocab is not None
    assert trained_w2v_model.vocab.size > 0  # Vocabulary should not be empty

    assert (
        trained_w2v_model.embeddings is not None
    )  # Ensure embeddings are created after training


def test_w2v_embedding(trained_w2v_model):
    embeddings = trained_w2v_model.get_embeddings(
        ["hello", "world", "fastvec", "is", "great"]
    )

    assert embeddings is not None
    assert len(embeddings) == 5  # Number of words in vocab
    assert len(embeddings[0]) == trained_w2v_model.embedding_dim  # Embedding dimension
