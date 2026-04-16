# Tweet Emotion Recognition with TensorFlow

This repository documents a multiclass natural language processing project that classifies tweets into discrete emotional categories using TensorFlow and recurrent neural networks. The work follows a guided deep learning exercise, but the implementation itself is structured as a complete text-classification pipeline: dataset ingestion, tokenization, sequence preprocessing, bidirectional LSTM modeling, training, and evaluation.

The model is trained to predict one of six emotions from tweet text:

- `sadness`
- `joy`
- `love`
- `anger`
- `fear`
- `surprise`

## Project Overview

The project frames tweet emotion recognition as a supervised multiclass classification problem in NLP. Short social-media texts are converted into padded integer sequences, passed through an embedding layer, modeled with stacked bidirectional LSTMs, and finally classified with a softmax output layer.

The notebook implementation focuses on building an end-to-end TensorFlow workflow for text classification rather than using a pretrained transformer. That makes the project a strong example of sequence modeling with classic deep learning components in Keras, especially for learning how tokenization, padding, recurrent layers, and label handling fit together in practice.

## Dataset

The project uses the Hugging Face `emotion` dataset, loaded in the notebook with:

```python
dataset = datasets.load_dataset('emotion')
```

The dataset is already partitioned into training, validation, and test splits:

| Split | Number of Samples |
| --- | ---: |
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

Each example contains:

- `text`: the tweet content
- `label`: the integer-encoded emotion class

Although the labels are already numeric in the dataset, the notebook also derives the class names from `train.features['label'].names` and builds dictionaries for readable class-to-index and index-to-class conversion. This is used later for interpretation and prediction display.

## Tech Stack

The notebook uses the following libraries and tools:

- TensorFlow / Keras for preprocessing, model construction, training, and evaluation
- NumPy for numeric array handling
- Matplotlib for plotting training curves and label/length distributions
- Hugging Face `datasets` for loading the `emotion` benchmark dataset
- Python `random` for sampling individual test examples
- `sklearn.metrics.confusion_matrix` for normalized confusion-matrix visualization

The recorded notebook output shows:

- TensorFlow version: `2.19.0`

One setup cell installs and imports Hugging Face's legacy `nlp` package:

```python
!pip install nlp
import nlp
```

However, the actual dataset loading logic in the notebook uses the modern `datasets` package through `datasets.load_dataset('emotion')`. This is worth noting because older guided materials sometimes retain the earlier package name even when the active workflow uses `datasets`.

## Data Preparation and Preprocessing Pipeline

The preprocessing flow in the notebook proceeds in a clear sequence:

1. Load the dataset and extract the `train`, `validation`, and `test` splits.
2. Use a helper function, `get_tweet(data)`, to separate tweet text and labels from each split.
3. Fit a Keras `Tokenizer` on the training tweets.
4. Convert text into integer token sequences.
5. Inspect tweet-length distribution with a histogram.
6. Pad and truncate sequences to a fixed maximum length.
7. Prepare labels for training and evaluation.

### Text Extraction

The helper below is used to unpack each dataset split into parallel lists of tweets and labels:

```python
def get_tweet(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels
```

This keeps the notebook simple and makes the downstream preprocessing steps easier to read.

### Tokenization

Tweet text is tokenized with Keras preprocessing utilities:

```python
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
```

Key tokenization settings:

- maximum vocabulary size: `10000`
- out-of-vocabulary token: `<UNK>`

This design limits the active vocabulary to the most frequent tokens while preserving a fallback representation for unseen words.

### Sequence Padding and Truncation

Before padding, the notebook measures tweet lengths by splitting each tweet on spaces and plotting a histogram. Based on that inspection, it fixes the sequence length at:

- `maxlen = 50`

The sequence conversion helper is:

```python
def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(
        sequences,
        truncating='post',
        padding='post',
        maxlen=maxlen
    )
    return padded
```

Padding choices used throughout the notebook:

- padding strategy: `post`
- truncation strategy: `post`
- sequence length: `50`

This ensures that every tweet becomes a fixed-size numeric input suitable for batching into the recurrent model.

### Label Handling

The class names are recovered from the dataset metadata:

```python
classes = train.features['label'].names
```

Which yields:

```text
['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
```

The notebook also creates:

- `class_to_index` for mapping class names to integer IDs
- `index_to_class` for decoding model predictions back into readable labels

Even though a helper named `names_to_ids` is defined, the actual train, validation, and test labels used for modeling are already numeric in the dataset, so the notebook ultimately relies on `np.array(labels)` rather than converting string labels.

## Model Architecture

The classifier is defined with Keras `Sequential` as:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
])
```

Architecture summary:

| Layer | Purpose |
| --- | --- |
| `Embedding(10000, 16)` | Learns dense vector representations for token IDs |
| `Bidirectional(LSTM(20, return_sequences=True))` | Captures contextual sequence information in both directions and preserves the full sequence for the next recurrent layer |
| `Bidirectional(LSTM(20))` | Aggregates the sequence into a final contextual representation |
| `Dense(6, activation='softmax')` | Produces class probabilities across the six emotion categories |

The model is compiled with:

```python
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

Important implementation note: the notebook includes a `model.summary()` cell, but the captured summary output shows the layers as unbuilt with zero parameters. For a technical description of the project, the architecture should therefore be taken from the source code definition above rather than from that unbuilt summary snapshot.

## Training Configuration

Training uses the padded training sequences and the validation split prepared with the same tokenizer and padding settings. The notebook trains with:

- loss function: `sparse_categorical_crossentropy`
- optimizer: `adam`
- metric: `accuracy`
- maximum epochs: `20`
- callback: `EarlyStopping(monitor='val_accuracy', patience=2)`

Training is launched with:

```python
h = model.fit(
    padded_train_seq, train_labels,
    validation_data=(val_seq, val_labels),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)
```

Although the epoch budget is 20, the recorded notebook run stops after 14 epochs due to early stopping behavior.

## Evaluation and Observed Results

The notebook evaluates the model in several complementary ways:

- visualization of training and validation accuracy/loss through `show_history(h)`
- quantitative evaluation on the held-out test set
- single-example qualitative prediction inspection
- normalized confusion matrix over the test split

The recorded run shows:

| Metric | Observed Value |
| --- | ---: |
| Best validation accuracy | `0.9035` |
| Test accuracy | `0.8860` |
| Test loss | `0.4346` |
| Training stopped after | `14` epochs |

These values should be interpreted as notebook-observed results from the recorded run, not as guaranteed fixed benchmarks for every environment or rerun.

### Single-Sample Inference

For interpretability, the notebook selects a random test example, prints the original sentence and true label, then predicts the class for that single padded sequence. The predicted class is decoded with `index_to_class[np.argmax(p)]`, which makes the output easier to inspect than raw probability vectors.

### Confusion Matrix

Predictions for the entire test set are generated with:

```python
preds = np.argmax(model.predict(test_seq), axis=-1)
```

The notebook then visualizes a normalized confusion matrix with `sklearn.metrics.confusion_matrix(..., normalize='true')`. This is a useful diagnostic because it highlights which emotions are classified reliably and which classes are more likely to be confused with one another.

## Key Technical Takeaways

- The project demonstrates a complete recurrent NLP workflow in TensorFlow without relying on pretrained language models.
- Tokenization and sequence standardization are central to making raw tweet text compatible with neural sequence models.
- Bidirectional LSTMs are used to model both forward and backward context, which is helpful for sentiment and emotion-bearing language patterns.
- Sparse categorical loss is an appropriate choice because the labels are stored as integer class IDs rather than one-hot vectors.
- Combining quantitative metrics with qualitative inspection and a confusion matrix provides a more complete view of classifier behavior than accuracy alone.

## Conclusion

This project is a compact but technically rich example of tweet-level emotion recognition with TensorFlow. It covers the full path from raw text to evaluated neural classifier, including vocabulary control, padding strategy, label interpretation, recurrent modeling, early stopping, and post-training analysis. As a repository artifact, it serves both as an NLP learning project and as a clear reference implementation of multiclass text classification with Keras-based sequence models.
