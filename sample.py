import gensim
import gensim.downloader


glove_vectors = gensim.downloader.load('glove-twitter-25')

print(glove_vectors.most_similar('food'))
print(glove_vectors.most_similar('hashtag'))