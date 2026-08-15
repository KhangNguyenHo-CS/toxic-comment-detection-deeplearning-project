# --------------------------------------
# Install Dependencies and Bring in Data
# --------------------------------------
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from matplotlib import pyplot as plt

file_path = os.path.join('data', 'train.csv')
df = pd.read_csv(file_path)

#Test data loading
#print(df.head()) #Show the first 5 rows of the dataframe
#print(df.tail()) #Show the last 5 rows of the dataframe

#print(df[df['toxic']==1].tail()) #Show the last 5 rows of the dataframe where the 'toxic' column is equal to 1

#print(df.iloc[6]['insult']) #Show the value of the 'insult' column for the 6th row of the dataframe
#print(df.iloc[6,2:]) #Show the values of the columns from the 2nd to the last column for the 6th row of the dataframe

# --------------------------------------
#
# ---------------------------------------
X = df['comment_text'] # The 'comment_text' column contains the text of the comments, which will be our input features (X).
Y = df.iloc[:, 2:].values # We use .values to convert the DataFrame into a NumPy array.
max_words = 20000 # The maximum number of words
vectorizer = TextVectorization(max_tokens=max_words, 
                               output_mode='int', 
                               output_sequence_length=1800) 

if not os.path.exists("vectorizer_vocab.txt"):
    print("Adapting vectorizer and saving vocabulary...")
    vectorizer.adapt(X.values) # Adapt the vectorizer to the text data (X)
    vocab = vectorizer.get_vocabulary() 
    with open("vectorizer_vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n") 
else:
    print("Loading existing vocabulary...")
    with open("vectorizer_vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]

vectorizer.set_vocabulary(vocab)

vectorized_text = vectorizer(X.values)

# print(len(X)) # Show the number of comments in the dataset
# print(vectorized_text)

#MCSHBAP - map, cache, shuffle, batch, prefetch from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, Y)) # Create a TensorFlow dataset from the vectorized text and labels (Y)
dataset = dataset.cache() # Cache the dataset in memory to improve performance during training.
dataset = dataset.shuffle(160000)  # Shuffle the dataset
dataset = dataset.batch(16) # Batch the dataset into batches of 16 samples. 
dataset = dataset.prefetch(8) # Prepare the dataset for training by prefetching 8 batches
                             # help bottlenecks 

#print(dataset.as_numpy_iterator().next()) # Show the first batch of data from the dataset as a NumPy array.
# batch_X, batch_Y = dataset.as_numpy_iterator().next() # Get the first batch of data from the dataset as NumPy arrays
# print(batch_X) # Show the vectorized text for the first batch
# print(batch_Y) # Show the labels for the first batch
# print(batch_X.shape) # Show the shape of the vectorized text for the first batch
# print(batch_Y.shape) # Show the shape of the labels for the first batch

train = dataset.take(int(len(dataset)*0.7)) # Take the first 70% of the dataset for training
valid = dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*0.2)) # Skip the first 70% and take the next 20% for validation
test = dataset.skip(int(len(dataset)*0.9)).take(int(len(dataset)*0.1)) # Skip the first 90% and take the last 10% for testing

# (train.as_numpy_iterator().next())

# --------------------------------------
# CREATE MODEL
# --------------------------------------

model_path = "toxic_model.keras"

if os.path.exists(model_path):
    print("Loading trained model...")
    model = load_model(model_path)
    print("Model loaded successfully")
else:
    print("Training model")
    print("Save into: " , model_path)
    model = Sequential()
    # Create the embedding layer
    model.add(Embedding(max_words+1, 32)) # Add an embedding layer to the model with a vocabulary size of max_words + 1 and an output dimension of 128.
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(32, activation='tanh'))) # Add a bidirectional LSTM layer with 64 units and return sequences set to True.
    # Feature extractor Fully connected layers (hidden layers)
    model.add(Dense(128, activation='relu')) # Add a dense layer with 128 units and ReLU activation.
    model.add(Dense(256, activation='relu')) # Add a dense layer with 256 units and ReLU activation.
    model.add(Dense(128, activation='relu')) # Add a dense layer with 128 units and ReLU activation.
    # Final layer (output layer)
    model.add(Dense(6, activation='sigmoid')) # Add a dense output layer with 6 units and sigmoid activation for multi-label classification.

    model.compile(loss = 'BinaryCrossentropy', optimizer = 'Adam')
    model.summary()

    history = model.fit(train, epochs = 15, validation_data = valid)

    model.save(model_path)
    print("Model train successfully")

# For ploting the training history

#print("Evaluating model...")
#history = model.fit(train, epochs = 1, validation_data = valid)
#print(history.history)
#plt.figure(figsize=(12, 6))
#pd.DataFrame(history.history).plot()
#plt.show()

# ---------------------------------------

input_text = vectorizer("I love you so much") 
print(input_text)
print(df.columns[2:])
batch = test.as_numpy_iterator().next()
batch_X, batch_Y = batch
#print(model.predict(np.array([input_text])))
#print(model.predict(batch_X))
#print((model.predict(batch_X) > 0.5).astype(int))
#res = model.predict(np.expand_dims(input_text, axis=0))
#res = model.predict(batch_X)
#print(res)

#Evaluate the Model
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    #unpack the batch
    x_true, y_true = batch

    #Make a prediction
    yhat = model.predict(x_true)


    #Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


print(f'''Precision: {pre.result().numpy()},
      Recall: {re.result().numpy()}, 
      Accuracy: {acc.result().numpy()}''')