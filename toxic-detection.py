# --------------------------------------
# Install Dependencies and Bring in Data
# --------------------------------------
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization

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
print(vectorizer('Fuck you Khang, you are the worse'))

