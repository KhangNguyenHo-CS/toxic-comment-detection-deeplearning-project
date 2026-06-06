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
Y = df.iloc[:, 2:].values # ..
max_words = 20000 # The maximum number of words
vectorizer = TextVectorization(max_tokens=max_words, 
                               output_mode='int', 
                               output_sequence_length=1800) 

#vectorizer.adapt(X.values) # Adapt the vectorizer to the text data (X)
