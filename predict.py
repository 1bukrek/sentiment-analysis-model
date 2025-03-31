import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

from termcolor import colored

# parameters
maxlen = 200

# 1. loading the trained model
model = load_model("imdb_sentiment_model.h5")
print(colored("Model loaded successfuly.", "green"))

# 2. loading the word index
word_index = imdb.get_word_index()

# 3. predict function
def predict_sentiment(text):
    tokens = [word_index.get(word, 0) for word in text.split()]
    tokens_padded = pad_sequences([tokens], maxlen=maxlen)
    prediction = model.predict(tokens_padded)[0, 0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment

# 4. example predicts
while True:
    user_input = input(colored("Enter a sentence (type 'exit' to exit): ", "light_green"))
    if user_input.lower() == "exit":
        break
    print(colored(f"Predict: {predict_sentiment(user_input)}", "light_blue"))
