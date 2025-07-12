from fastvec import Word2Vec, Doc2Vec, simple_preprocessing


def test_word2vec():
    small_corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog jumps over the quick brown fox",
        "the quick brown fox is quick and clever",
        "the lazy dog is lazy and slow",
    ]
    tokens = simple_preprocessing(small_corpus, deacc=False)
    print("Tokens:", tokens.tokens)
    model = Word2Vec(embedding_dim=64, epochs=10)
    model.train(tokens)
    print("Word2Vec Model trained successfully.")

    embeddings = model.get_embeddings(["quick", "lazy", "fox", "dog"])
    print("Embeddings for 'quick', 'lazy', 'fox', 'dog':", embeddings)


def test_doc2vec():
    small_corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog jumps over the quick brown fox",
        "the quick brown fox is quick and clever",
        "the lazy dog is lazy and slow",
    ]
    tokens = simple_preprocessing(small_corpus, deacc=False)
    print("Tokens:", tokens.tokens)
    model = Doc2Vec(embedding_dim=64, epochs=10)
    model.train(tokens)
    print("Doc2Vec Model trained successfully.")

    embeddings = model.get_embeddings(["the quick brown fox jumps over the lazy dog"])
    print("Embeddings for 'quick', 'lazy', 'fox', 'dog':", embeddings)


def main():
    # Test Word2Vec
    # print("Testing Word2Vec...")
    # test_word2vec()

    print("Testing Doc2Vec...")
    test_doc2vec()


if __name__ == "__main__":
    main()
