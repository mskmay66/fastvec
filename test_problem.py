from fastvec import Word2Vec, simple_preprocessing, Vocab, Embedding, Builder, Tokens

def main():
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
    print("Model trained successfully.")

if __name__ == "__main__":
    main()