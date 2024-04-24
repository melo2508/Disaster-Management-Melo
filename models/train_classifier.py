import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import joblib


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    
        """Description of the Function:
    Loading data from SQLite database
        
    Parameters:
        database_filepath: Database file path.
       
    Returns:
       X: Network input
       y: Network output
       catergory_names: Catergories names
    """
    
        engine = create_engine('sqlite:///'+database_filepath)
        df = pd.read_sql_table('InsertTableName',engine)
        #df
        X = df['message'] 
        y = df.iloc[:, 4:]
        category_names = y.columns
        
        return X, y, category_names

def tokenize(text):

    """Description of the Function:
     
       Tokenizing the text data
    
     1- Split text into tokens.
     2- For each token: lemmatize, normalize case, 
    and strip leading and trailing white space.
     3- Return the tokens in a list.
    
    Parameters:
    text: Text string
       
    Returns:
       clean_tokens: Tokenized text
    
    """

    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens



def build_model():
    


    pipeline = Pipeline([
        ('features', FeatureUnion([
    
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
    
            
        ])),
    
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
                
                 
    parameters = {  
        
                    'features__text_pipeline__vect__max_features': (None, 5000, 10000),
                    'features__text_pipeline__tfidf__use_idf': (True, False),
                    'classifier__estimator__n_estimators': [100,500],
                    'classifier__estimator__max_depth': [None, 80, 100],
                    'classifier__estimator__max_features':[ None ,'auto', 20, 30]
                }
    
    
       
    cv = GridSearchCV(pipeline, param_grid = parameters)
       
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    
    """Description of the Function:
     
       Evaluating the created model 
        
     1- Split text into tokens.
     2- For each token: lemmatize, normalize case, 
        and strip leading and trailing white space.
     3- Return the tokens in a list.
        
     
        
    Parameters:
        model: The created model
        X_test: Test input data
        Y_test: Test output data
        category_names: Categories names
       
    """
    
    y_pred_tuned = model.predict(X_test)
    for col in category_names:
        
        print(col, classification_report(Y_test[col].values, y_pred_tuned[:,int(np.where(category_names==col)[0])]))



def save_model(model, model_filepath):
    
    """Description of the Function:
    Saving the creatde model
    
    Parameters:
    model: created model
    model_filepath: file path 
       
    """
      
    joblib.dump(model, model_filepath)
    
    

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