import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from DisasterResponse", con=engine)
    X = df["message"]
    Y = df[['related', 'request', 'offer',
            'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'child_alone', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity',
            'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
            'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
            'other_weather', 'direct_report']]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if not w in stop_words]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        # 'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [50],
        # 'clf__estimator__min_samples_split': [2, 3, 4],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = cv.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(y_test[category_names[i]].values, y_pred[:, i]))


def save_model(model, model_filepath):
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