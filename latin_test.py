from cltk import NLP
import numpy as np
import pandas as pd
import sklearn
import scipy
import nltk

latin = NLP(language_code="lat")

text = "Gallia est omnis divisa in partes tres."

doc = latin.analyze(text)

print("Tokens:")
for word in doc.words:
    print(word.string)

print("\nEnvironment ready.")