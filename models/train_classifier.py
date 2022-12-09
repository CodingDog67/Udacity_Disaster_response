import sys
import pickle
import warnings


import nltk
import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

nltk.download('omw-1.4')
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
   """
   input database_filepath as string
   
   output:
   X = training data 
   Y = labels
   category_names = classes
   """
   #assert os.path.exists(database_filepath)
   #conn = sqlite3.connect("../" + database_filepath)
   #table_data = pd.read_sql_table('database_filepath', conn)

   engine = create_engine('sqlite:///' + "../" + database_filepath)
   table_data = pd.read_sql_table('DisasterResponse_table', engine)

   X = table_data["message"]
   Y = table_data.iloc[:, 4:]
   category_names = Y.columns 

   return X, Y, category_names


def tokenize(text):

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlreplace")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(clf =AdaBoostClassifier()):
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf))

    ])

    params = {'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 15]}

    cv = GridSearchCV(pipeline, param_grid=params, verbose=2, n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Takes the trained model and applies it on the test set
    
    Output: The predictions of the testset, will print the scores for each class
    
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        pred_Y_test = model.predict(X_test)

        for iter, category in enumerate(category_names):
            print('---{}---'.format(category.upper()))
            print(classification_report(Y_test[category].to_numpy(), pred_Y_test[:, iter]))


def save_model(model, model_filepath):
    ''''
    input: the trained model
    output: saving the model as a pickle file to filepath
    '''

    Pkl_Filename = "classifier.pkl"

    if os.path.exists(model_filepath):
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_filepath = '../' + model_filepath
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath = sys.argv[2:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
