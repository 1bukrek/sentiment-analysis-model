# Sentiment Analysis with LSTM (IMDB Dataset)

This project is a sentiment analysis model using TensorFlow and LSTM to classify IMDB movie reviews as positive or negative.

## Model Details

-   Uses an LSTM network with embedding layers. More detailed explanations about the model are given in the comments in the files.
-   Trained on the IMDB dataset (preprocessed movie reviews).
-   Output is binary (0 = Negative, 1 = Positive).
-   Model is saved as `imdb_sentiment_model.h5`.

## Installation

1. Clone the repository:

```
git clone https://github.com/1bukrek/sentiment-analysis-model.git
cd sentiment-analysis
```

2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install tensorflow numpy termcolor
```

## Training the Model

To train the model, run: `python train.py`. This will load the IMDB dataset, train an LSTM model and will save the trained model as imdb_sentiment_model.h5.

##  Making Predictions

Once the model is trained, use `python predict.py` command. It will prompt you to enter a sentence and classify it as positive or negative.

### Notes

-   If accuracy is low, you can try increasing epochs or tuning hyperparameters.

-   Ensure train.py is run first before using predict.py.
