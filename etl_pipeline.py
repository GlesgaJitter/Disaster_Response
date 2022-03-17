# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv("messages.csv")
messages.head()


# load categories dataset
categories = pd.read_csv("categories.csv")
categories.head()

# merge datasets
df = messages.merge(categories, how='inner', left_on='id', right_on='id')
df.head()

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';',expand=True)
categories.head()

# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
category_colnames = row.str[:-2]
#category_colnames


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

# convert category values to 0 or 1
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1:]

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

    categories.replace(to_replace=2, value=1, inplace=True)
categories.head()

print(categories.related.unique())

# drop the original categories column from `df`
df.drop('categories', axis=1, inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)

# check number of duplicates
df_drop_dup = df.drop_duplicates()
df.shape[0] - df_drop_dup.shape[0]

# Save the clean dataset into an sqlite database.
engine = create_engine('sqlite:///messages_cleaned.db')
df.to_sql('messages_cleaned', engine, index=False)
