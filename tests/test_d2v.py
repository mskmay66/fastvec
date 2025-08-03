from fastvec import Doc2Vec, Tokens
import pytest


@pytest.fixture
def doc2vec_model():
    """Fixture to create a Doc2Vec model for testing."""
    return Doc2Vec(embedding_dim=3, epochs=2, inference_epochs=1)


@pytest.fixture
def sample_tokens():
    """Fixture to provide sample tokens for testing."""
    return Tokens(tokens=[["hello", "world"], ["fast", "vector"]])


def test_doc2vec_build_vocab(doc2vec_model, sample_tokens):
    """Test building vocabulary from sample tokens."""
    doc2vec_model.build_vocab(sample_tokens.flatten())
    assert doc2vec_model.vocab is not None
    assert len(doc2vec_model.vocab) > 0


def test_doc2vec_build_training_set(doc2vec_model, sample_tokens):
    """Test building the training set from sample tokens."""
    doc2vec_model.build_vocab(sample_tokens.flatten())
    training_set = doc2vec_model.build_training_set(sample_tokens.tokens, window_size=2)
    assert training_set is not None
    assert len(training_set) > 0


def test_doc2vec_train(doc2vec_model, sample_tokens):
    """Test training the Doc2Vec model."""
    doc2vec_model.train(sample_tokens, window_size=2)

    print(doc2vec_model.embeddings.vectors)
    print(doc2vec_model.vocab.valid_ids)
    assert doc2vec_model.embeddings is not None
    assert len(doc2vec_model.embeddings) > 0


def test_doc2vec_inference(doc2vec_model, sample_tokens):
    """Test inference with the Doc2Vec model."""
    doc2vec_model.build_vocab(sample_tokens.flatten())
    doc2vec_model.train(sample_tokens, window_size=2)

    embeddings = doc2vec_model.get_embeddings(sample_tokens.tokens[0])
    assert embeddings is not None
    assert len(embeddings) == len(sample_tokens.tokens[0])
    print(embeddings)
    assert all(isinstance(embedding, list) for embedding in embeddings)
