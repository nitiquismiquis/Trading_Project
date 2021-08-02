# Disaster Response Pipeline Project

<h1>Instructions</h1>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<h2>Motivation</h2>

Create an online dashboard capable of categorising text messages in terms of disaster reponse using Machine Learning.

<h3>Results</h3>
   
   - All messages are categorised by the following predictors:
    
    * 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security',
       'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather'
    
	* The results from the model are the following (classification_report):
							 precision recall    f1-score  support

               request       0.83      0.48      0.61      1519
                 offer       0.00      0.00      0.00        39
           aid_related       0.74      0.69      0.71      3591
          medical_help       0.65      0.12      0.20       702
      medical_products       0.62      0.12      0.21       444
     search_and_rescue       0.70      0.09      0.16       231
              security       0.33      0.01      0.01       147
              military       0.65      0.10      0.18       267
           child_alone       0.00      0.00      0.00         0
                 water       0.88      0.42      0.57       581
                  food       0.83      0.65      0.73       969
               shelter       0.86      0.44      0.58       765
              clothing       0.71      0.21      0.32       138
                 money       0.82      0.04      0.08       205
        missing_people       0.50      0.01      0.02       101
              refugees       0.54      0.07      0.13       275
                 death       0.82      0.20      0.32       397
             other_aid       0.53      0.05      0.10      1164
infrastructure_related       0.06      0.00      0.00       579
             transport       0.59      0.11      0.19       409
             buildings       0.77      0.16      0.27       434
           electricity       0.80      0.04      0.08       184
                 tools       0.00      0.00      0.00        54
             hospitals       0.00      0.00      0.00        93
                 shops       0.00      0.00      0.00        40
           aid_centers       0.00      0.00      0.00       105
  other_infrastructure       0.00      0.00      0.00       404
       weather_related       0.85      0.71      0.77      2457
                floods       0.91      0.45      0.61       707
                 storm       0.79      0.58      0.66       846
                  fire       1.00      0.04      0.08        94
            earthquake       0.90      0.81      0.85       818
                  cold       0.81      0.11      0.20       189
         other_weather       0.52      0.05      0.10       455

           avg / total       0.71      0.42      0.49     19403

<h2>Libraries Used</h2>
    * numpy
    * pandas
    * sqlalchemy
	* nltk
	* pickle
    * sklearn
   
<h2>Files</h2>

    * disaster_categories.csv
    * disaster_messages.csv
    * process_data.py
	* train_classifier.py
	* run.py
	   * go.html
	   * master.html
    * README.md

<h2>acknowledgements</h2>
 
    * https://data.worldbank.org/
    * Anaconda3
    * GitHub Pages
	* sqlalchemy
	* nltk
	* pickle
    * Scikit-learn
   
<h2>Contacts</h2>

   Rodrigo - nitiquismiquis@hotmail.com
   Project Link: https://github.com/nitiquismiquis/Disaster_Resp_pipelines.git