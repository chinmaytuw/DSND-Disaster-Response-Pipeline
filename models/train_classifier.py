import sys
import pandas as pd 
import numpy as np
import pickle
import nltk
from sqlalchemy import create_engine


nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier


def load_data(database_filepath):
    ''' 
    Script to load the data 

    Args:
    	database_filepath: path of the database file

    Returns:
    	X: Feature message
    	Y: Target Values
    	category_names: names of categories available in tbe data
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    category_names = Y.columns
    return X , Y , category_names


def tokenize(text):
    ''' 
    Function to tokenize text

    Args:
    	text: input text

    Returns:
    	clean_tokens: returns cleaned tokens 
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    ''' 
    Function that builds the model and returns cross GridSearchCV object

    Args:
        None

    Returns:
        cv: GridSearchCV object
    '''
    pipeline  = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=0))))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    Function for evaluating the model

    Args:
    	model: model object
    	X_test: test features
    	Y_test: test labels
    	category_names: list of available categories

    Returns:
    	None
    '''
    Y_pred = model.predict(X_test)
    for i in range(36):
        print("classification report for " + Y_test.columns[i]
              ,'\n', classification_report(Y_test.values[:,i],Y_pred[:,i])
              ,'\n accuracy:', accuracy_score(Y_test.values[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    ''' 
    Function to export model as pickle object

    Args:
    	model: model object
    	model_filepath: path for model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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








