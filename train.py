# import libraries
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
import lightgbm


# load data from database
engine = create_engine('sqlite:///messages_cleaned.db')
df = pd.read_sql_table('messages_cleaned', 'sqlite:///messages_cleaned.db')
X = df.message
#X = df.message.values
#y = df.columns[4:]
y = df.loc[:,"related":"direct_report"]
print(y.shape)
print(y.related.unique())

def tokenize(text):

    """
    comments
    """


    # make lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    words = word_tokenize(text)
    #words = [w for w in words if w not in stopwords.words("english")]
    lemmatiser = WordNetLemmatizer()

    clean_tokens = []
    for word in words:
        clean_word = lemmatiser.lemmatize(word).lower().strip() #
        clean_tokens.append(clean_word)

    return clean_tokens


# Build ML Pipeline
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
])

# train classifier
pipeline.fit(X, y)

# predict on test data
#y_pred = pipeline.predict(X_test)

# display results
#display_results(y_test, y_pred)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)
