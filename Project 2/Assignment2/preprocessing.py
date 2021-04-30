import requests
import re
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()
ROOT.withdraw()

# the input dialog
user_input = int(simpledialog.\
    askstring(title='Fine-tune BERT',
              prompt='Enter following for the respective tasks\n1 --> '
              'hate\n2 --> offensive\n3 --> sentiment'))

# dictionary linking input with tasks
dict = {'hate': 1,'offensive': 2, 'sentiment': 3}
datasets = ['hate', 'offensive', 'sentiment']

#Loading DATA
for data in dict.values():
    if data == user_input:
        task = list(dict.keys())[data-1]
        print(task)
        text = requests.get('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/'+task+'/train_text.txt').text
        label = requests.get('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/'+task+'/train_labels.txt').text
        val_text = requests.get('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/'+task+'/val_text.txt').text
        val_label = requests.get('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/'+task+'/val_labels.txt').text
        text_test = requests.get('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/'+task+'/test_text.txt').text
        label_test = requests.get('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/'+task+'/test_labels.txt').text
        print('*'*20+'Files Loaded from GITHUB'+'*'*20)

# Formatting into DATAFRAME
def process(label, text):
    tag = []
    for sent in label.split("\n"):
        try:
            tag.append(int(sent))
        except ValueError:
            pass

    tweet = []
    for text in text.split('\n'):
        try:
            tweet.append(text)
        except ValueError:
            pass

    data = {'tweet': tweet[:-1], 'tag': tag}
    df = pd.DataFrame(data)
    df['class'] = df.tag.apply(lambda x: 'not-hate' if x == 0 else 'hate')
    return df

# Function to clean text
def cleaner(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.replace("user", "")
    return tweet

# Applying cleaner function and deleting words with length less than or equal to 2
def cleanup(df):
    train_cleaned = df['tweet'].apply(cleaner)
    df['tweet'] = train_cleaned.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    return df

# Preparing train, test, validation dataframe for BERT processing
def cleaned_df():
    df = process(label, text)
    df_val = process(val_label, val_text)
    df_test = process(label_test, text_test)

    train_cleaned = cleanup(df)
    val_cleaned = cleanup(df_val)
    test_cleaned = cleanup(df_test)

    lst = [train_cleaned, val_cleaned]
    train_cleaned = pd.DataFrame(np.concatenate(lst), columns=val_cleaned.columns)
    print('*'*20+'TEXT PRE-PROCESSING DONE'+'*'*20)
    return train_cleaned, val_cleaned, test_cleaned


