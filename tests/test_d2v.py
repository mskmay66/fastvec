from fastvec import Doc2Vec, Doc2VecDataset, Builder, simple_preprocessing
import pytest
import torch


@pytest.fixture(scope="module")
def d2v_model():
    d2v = Doc2Vec(embedding_dim=64, epochs=10)
    return d2v


@pytest.fixture(scope="module")
def trained_d2v_model(d2v_model):
    corpus = ["hello world", "fastvec is great"]
    tokens = simple_preprocessing(corpus, deacc=True)
    d2v_model.train(tokens, window_size=2)
    return d2v_model


def test_d2v_creation(d2v_model):
    assert d2v_model.embedding_dim == 64
    assert d2v_model.epochs == 10
    assert d2v_model.vocab is None
    assert d2v_model.embeddings is None
    assert d2v_model.input_encoder is not None
    assert d2v_model.target_encoder is not None
    assert d2v_model.document_encoder is not None
    assert d2v_model.activation is not None
    assert d2v_model.input_encoder.weight.shape == (64, 1)
    assert d2v_model.target_encoder.weight.shape == (64, 1)
    assert d2v_model.document_encoder.weight.shape == (64, 1)


def test_d2v_forward(d2v_model):
    doc_id = torch.tensor([[1], [2]], dtype=torch.float32)
    input_words = torch.tensor([[3], [4]], dtype=torch.float32)
    target_words = torch.tensor([[5], [6]], dtype=torch.float32)

    output = d2v_model.forward(doc_id, input_words, target_words)

    assert output.shape == (2,)
    assert output.dtype == torch.float32  # Output should be of float type


def test_d2v_build_vocab(d2v_model):
    corpus = ["hello", "world", "fastvec", "is", "great"]
    d2v_model.build_vocab(corpus)

    assert d2v_model.vocab is not None
    assert d2v_model.vocab.size > 0  # Vocabulary should not be empty
    assert "hello" in d2v_model.vocab.word_to_id
    assert "world" in d2v_model.vocab.word_to_id
    assert "fastvec" in d2v_model.vocab.word_to_id
    assert "is" in d2v_model.vocab.word_to_id
    assert "great" in d2v_model.vocab.word_to_id


def test_d2v_dataset(d2v_model):
    corpus = ["hello world", "fastvec is great"]
    tokens = simple_preprocessing(corpus, deacc=True)
    d2v_model.build_vocab(tokens.flatten())
    builder = Builder(tokens.tokens, d2v_model.vocab, 3)
    examples = builder.build_d2v_training()

    dataset = Doc2VecDataset(examples)

    assert len(dataset) == len(examples)

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "doc_id" in sample
    assert "input_words" in sample
    assert "target_words" in sample
    assert "label" in sample
    assert sample["doc_id"].dtype == torch.float32
    assert sample["input_words"].dtype == torch.float32
    assert sample["target_words"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32


def test_d2v_training_set(d2v_model):
    corpus = ["hello world", "fastvec is great"]
    tokens = simple_preprocessing(corpus, deacc=True)
    d2v_model.build_vocab(tokens.flatten())
    training_set = d2v_model.build_training_set(tokens.tokens, window_size=3)

    assert isinstance(training_set, torch.utils.data.DataLoader)
    assert len(training_set.dataset) > 0  # Training set should not be empty
    for batch in training_set:
        assert "doc_id" in batch
        assert "input_words" in batch
        assert "target_words" in batch
        assert "label" in batch
        assert batch["doc_id"].dtype == torch.float32
        assert batch["input_words"].dtype == torch.float32
        assert batch["target_words"].dtype == torch.float32
        assert batch["label"].dtype == torch.float32


def test_d2v_training(trained_d2v_model):
    assert trained_d2v_model.vocab is not None
    assert trained_d2v_model.vocab.size > 0  # Vocabulary should not be empty

    assert (
        trained_d2v_model.embeddings is not None
    )  # Ensure embeddings are created after training


def test_d2v_inference(trained_d2v_model):
    assert trained_d2v_model.embeddings is not None  # Ensure embeddings are learned
    assert len(trained_d2v_model.embeddings) > 0  # Ensure embeddings are not empty

    output = trained_d2v_model.get_embeddings(["feastvec is swell!"])
    assert output is not None
    assert len(output) == 1  # One document should return one embedding


def test_d2v_embedding(trained_d2v_model):
    embeddings = trained_d2v_model.get_embeddings(
        ["hello", "world", "fastvec", "is", "great"]
    )

    assert embeddings is not None
    assert len(embeddings) == 5  # Number of words in vocab
    assert len(embeddings[0]) == trained_d2v_model.embedding_dim  # Embedding dimension
