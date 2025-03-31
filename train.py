import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

from termcolor import colored

# parameters
max_features = 10000  # most frequently used words (top 10k words)
maxlen = 200  # max length of the comments (padding/truncating length)
batch_size = 64
epochs = 5

# 1. loading the dataset and getting ready
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad the sequences to ensure uniform input size for the model
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 2. identifying the model/layers:

# embedding layer: embedding layer will convert words to numbers and then numeric vectors.
# input_dim=max_features: this sets the maximum word count that model is going to work with.
# output_dim=128: this specifies how many dimensions each word will be converted into a vector.

# LSTM (long short-term memory) layer: LSTM performs learning on data by taking into account the time series or the order of words.
# 128: this specifies the output units (neurons) of the LTSM layer (128 unit memory).
# dropout=0.2: this specifies the rate at which some connections will be randomly closed during training. this is a precaution for overfitting problem.
# recurrent_dropout=0.2: this is the dropout rate on the internal (recurrent) connections of the LSTM.

# output layer: the model only predicts 0 or 1, so there is only one unit in the output layer.
# the sigmoid function converts the output value into a probability value between 0 and 1.

model = Sequential([
    Embedding(input_dim=max_features, output_dim=128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# 3. compiling the model: compiling gets the model ready for training.

# 1. loss function(loss='binary_crossentropy'): the loss function measures how much the model got wrong.
# "binary_crossentropy" is used because the output will be 0 or 1.
# 2. optimization algorithm(optimizer=Adam()): the optimization algorithm updates the weights to improve the learning process of the model. (Adaptive Moment Estimation)
# 3. performance metrics(metrics=['accuracy']): used to see the change in accuracy during training.

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

# 4. adding checkpoint callback to save the best model during training
checkpoint = ModelCheckpoint("imdb_sentiment_model.h5", 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

# 5. training the model
model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint])

# 5. saving the model for later usage
model.save("sentiment_model.keras")
print(colored("Model saved successfuly: 'sentiment_model.keras'", "green"))