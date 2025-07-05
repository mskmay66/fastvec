# -*- coding: utf-8 -*-
from .fastvec import *
from .word2vec import Word2Vec

__doc__ = fastvec.__doc__
if hasattr(fastvec, "__all__"):
    __all__ = fastvec.__all__
    __all__.append("Word2Vec")
