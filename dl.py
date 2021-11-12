import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import array
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.preprocessing.text import Tokenizer

tweets = pd.read_csv('data/dl.csv')
tweets.shape

X = []
tw = list(tweets['tweet'])
for t in tw:
    X.append(t)

y = tweets['class']
y = np.array(list(map(lambda x: 1 if x=="positive" else (0 if x=="neutral" else 0), y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

tok = Tokenizer(num_words=1000)
tok.fit_on_texts(X_train)

X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)

vocab = len(tok.word_index) + 1

length = 80

X_train = pad_sequences(X_train, padding='pre', maxlen=length)
X_test = pad_sequences(X_test, padding='pre', maxlen=length)

dict = dict()
glv = open('data/glove.txt', encoding="utf8")

for l in glv:
    spl = l.split()
    rem = spl[0]
    dims = asarray(spl[1:], dtype='float32')
    dict [rem] = dims
glv.close()

matrix = zeros((vocab, 100))
for rem, index in tok.word_index.items():
    vector = dict.get(rem)
    if vector is not None:
        matrix[index] = vector

model = Sequential()
layer = Embedding(vocab, 100, weights=[matrix], input_length=length , trainable=False)
model.add(layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
results_1 = model.fit(X_train, y_train, batch_size=100, epochs=8, verbose=0, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy:", score[1] * 100)

plt.plot(results_1.history['acc'])
plt.plot(results_1.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()
plt.plot(results_1.history['loss'])
plt.plot(results_1.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

model = Sequential()
embedding_layer = Embedding(vocab, 100, weights=[matrix], input_length=length , trainable=False)
model.add(embedding_layer)
model.add(Conv1D(100, 3, activation='sigmoid'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
results_2 = model.fit(X_train, y_train, batch_size=100, epochs=8, verbose=0, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Accuracy:", score[1] * 100)

plt.plot(results_2.history['acc'])
plt.plot(results_2.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()
plt.plot(results_2.history['loss'])
plt.plot(results_2.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()