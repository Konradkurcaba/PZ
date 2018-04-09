
import nltk

nltk.download('reuters')
from nltk.corpus import reuters


documents = reuters.fileids()
print(str(len(documents)) + " documents");
