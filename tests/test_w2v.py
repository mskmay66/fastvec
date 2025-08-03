from fastvec import Word2Vec, Tokens
import pytest


@pytest.fixture
def word2vec_model():
    """Fixture to create a Word2Vec model for testing."""
    return Word2Vec(embedding_dim=3, epochs=2, lr=0.01)


@pytest.fixture
def sample_tokens():
    """Fixture to provide sample tokens for testing."""
    return Tokens(tokens=[["hello", "world"], ["fast", "vector"]])


def test_word2vec_build_vocab(word2vec_model, sample_tokens):
    """Test building vocabulary from sample tokens."""
    word2vec_model.build_vocab(sample_tokens.flatten())
    assert word2vec_model.vocab is not None
    assert len(word2vec_model.vocab) > 0


def test_word2vec_build_training_set(word2vec_model, sample_tokens):
    """Test building the training set from sample tokens."""
    word2vec_model.build_vocab(sample_tokens.flatten())
    training_set = word2vec_model.build_training_set(
        sample_tokens.tokens, window_size=2
    )
    assert training_set is not None
    assert len(training_set) > 0


def test_word2vec_train(word2vec_model, sample_tokens):
    """Test training the Word2Vec model."""
    word2vec_model.train(sample_tokens, window_size=2)

    print(word2vec_model.embeddings.vectors)
    print(word2vec_model.vocab.valid_ids)
    assert word2vec_model.embeddings is not None
    assert len(word2vec_model.embeddings) > 0
