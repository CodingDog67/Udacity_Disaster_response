import sys

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    #reading in csv data and saving as pd frame
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # messages.head() check

    #merge the two into a single database
    data = pd.merge(messages, categories, on="id")

    return data

def clean_data(df):

    #1) splits the category column in invididual classes and assign proper column/class names
    clean_categories = df['categories'].str.split(';', expand=True)
    row = clean_categories.head(1).values.tolist()[0]
    class_names = [classes.split('-')[0] for classes in row]
    clean_categories.columns = class_names

    # clean categories
    for col_name in clean_categories:
        # remove the name and keep the label
        clean_categories[col_name] = clean_categories[col_name].str[-1].astype(int)

        # replace any label that is bigger 1 with 1
        # as there should be only 0 and 1 for belongs and don't belong to class
        #drop any column that has only one kind of label as it is useless
        unique_labels = np.unique(clean_categories[col_name])
        if len(unique_labels) == 1:
            clean_categories.drop(col_name, axis=1,inplace=True)
            continue
        if len(unique_labels) == 2:
            continue
        else:
            replace_these = np.delete(unique_labels, [0, 1])
            for val in replace_these:
                clean_categories[col_name].replace(val, 1)


    # drop original categories column
    df.drop(columns=['categories'], inplace=True, axis=1)

    # concat new clean categories with data
    df = pd.concat([df, clean_categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    assert df.duplicated().sum() == 0

    return df

def save_data(df, database_filename):
    """
    input : cleaned df
    output: "DisasterResponse" database
    """

    #database =  "../test.sqlite"
    #conn = sqlite3.connect(database)
    #df.to_sql(name=database_filename, con=conn)
    #conn.close()

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath = sys.argv[2:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()