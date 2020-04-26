# Disaster Response Pipeline Project

This project is an exercise provided by Udacity. It is divided into three sessions: the ETL pipeline, the machine learning pipeline, and the Flask Web App.

## The ETL pipeline 
To run ETL pipeline that cleans data and stores in database
- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  
## The machine learning pipeline 
To run ML pipeline that trains classifier and saves
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## The Flask Web app
Run the following command in the app's directory to run your web app.
- `python run.py`
- go to http://0.0.0.0:3001/
