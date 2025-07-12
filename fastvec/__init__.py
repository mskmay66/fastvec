# -*- coding: utf-8 -*-
# from .fastvec import *
# from .word2vec import Word2Vec
# from .doc2vec import Doc2Vec

# __doc__ = fastvec.__doc__
# if hasattr(fastvec, "__all__"):
#     __all__ = fastvec.__all__
#     __all__.append("Word2Vec")
#     __all__.append("Doc2Vec")

from .fastvec import Vocab, Embedding, Builder, Tokens
from .word2vec import Word2Vec
from .doc2vec import Doc2Vec

__all__ = ["Word2Vec", "Doc2Vec", "Vocab", "Embedding", "Builder", "Tokens"]
