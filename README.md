# Long-Short-Term-Memory-LSTM-Networks

## Overview
This project implements a **Long Short-Term Memory (LSTM)** network to perform sentiment analysis on the **IMDB movie reviews dataset**. The model classifies reviews as **positive** or **negative** based on textual content.

The dataset consists of **50,000** movie reviews, with an equal split between positive and negative reviews. The model processes and classifies text using an **LSTM-based deep learning approach**.

## Features
- Uses **LSTM** to capture long-term dependencies in text.
- Preprocesses text data with **word embedding** and **sequence padding**.
- Evaluates model performance using accuracy and loss plots.
- Predicts sentiment for new reviews.

## Prerequisites
Ensure you have the following dependencies installed:

- Python (>=3.7)
- NumPy
- Matplotlib
- TensorFlow / Keras

To install the required packages, run:

pip install numpy matplotlib tensorflow keras

## Dataset
The **IMDB dataset** contains 50,000 movie reviews labeled as **positive (1)** or **negative (0)**. The dataset is preprocessed to keep only the **10,000 most frequent words**, and each review is padded to a **fixed length of 500 words**.

## Installation and Setup
1. Clone the repository:

   git clone https://github.com/Satya-dev123/Long-Short-Term-Memory-LSTM-Networks.git

2. Navigate to the project directory:

   cd Long Short-Term Memory Networks.ipynb

3. Run the Jupyter Notebook or Python script:
   
   jupyter notebook / Run all cells

## Implementation Steps
1. **Load the IMDB dataset** with the most frequent 10,000 words.
2. **Pad sequences** to ensure all reviews have the same length.
3. **Convert labels** to categorical format for classification.
4. **Build an LSTM model** with an embedding layer, LSTM layer, and dense output layer.
5. **Train the model** using `categorical_crossentropy` loss and `adam` optimizer.
6. **Evaluate model performance** on the test dataset.
7. **Visualize accuracy and loss trends** using Matplotlib.
8. **Predict sentiment** on new test reviews.

## Example Code

# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

# Step 2: Load and preprocess the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Step 3: Pad the sequences to ensure they all have the same length
maxlen = 500
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Step 4: Convert labels to categorical
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Step 5: Build the LSTM model
model = models.Sequential()
model.add(layers.Embedding(input_dim=10000, output_dim=128, input_length=maxlen))
model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(2, activation='softmax'))

# Step 6: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Step 8: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Step 9: Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 10: Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 11: Predict sentiment for sample reviews
sample_reviews = x_test[:5]
predictions = model.predict(sample_reviews)

# Step 12: Display predictions
for i, prediction in enumerate(predictions):
    print(f"Review {i+1}: {'Positive' if np.argmax(prediction) == 1 else 'Negative'}")
    print("Prediction Probabilities:", prediction)
    print("-" * 50)

## Model Performance Metrics
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** Adam
- **Evaluation Metric:** Accuracy
- **Training & Validation Plots:** Accuracy and loss trends over epochs

## Visualizations
- **Accuracy Plot:** Tracks model performance across epochs.
- **Loss Plot:** Displays training vs. validation loss trends.

## Future Improvements
- Increase **LSTM layers** and add **bidirectional LSTM** for better feature extraction.
- Implement **attention mechanisms** to focus on important words.
- Use **pre-trained word embeddings** (e.g., GloVe or Word2Vec) for enhanced representation.
- Experiment with **hyperparameter tuning** for better accuracy.

## License
This project is licensed under the MIT License.
