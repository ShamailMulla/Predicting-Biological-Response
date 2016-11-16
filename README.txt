Project Description:	
    This project aims at enhancing targeted medical treatments for illnesses using Machine Learning. The project has 2 files	
    having input datasets. A training set and a test set. These are in csv file formats in the project folder.	
    This will be used create an algorithm which learns behaviour of biological molecules to a particular stimulus and 	
    then be able to predict how a similar molecule would predict in the future. The program analyses the test data provided and 	
    outputs the molecular behaviour for each molecule. The size of the test set is 2501 molecules and it's predictions	
    are stored it in 'Predictions.csv'.	
	
.pdf files	
	1) 'CAPSTONE PROJECT REPORT.pdf' contains a detailed report about this project
	
.py files	
	1) 'PredictingBiologicalResponse.py' is the startup file for the project
	2) 'PredictingBiologicalResponse_Analysis.py' performs all the computations and calculations on the data
	3) 'PredictingBiologicalResponse_Initialisations.py' loads the data sets
	4) 'PredictingBiologicalResponse_Visualisations.py' plots the data outputted by 'PredictingBiologicalResponse_Analysis.py' on graphs
	
.csv Files:	
	1) 'train.csv' contains the training datasets
	2) 'test.csv' contains the testing datasets
	3) 'benchmark.csv' contains the predicted probabilities benchmark values for test set (in test.csv)
	4) 'Features Information.csv' contains statistical information of the dataset provided
	5) 'Predictions.csv' contains the molecular response predicted by the model
	6) 'Prediction Probability.csv' contains the actual predictions probabilities made by the model for the test set
  	
Running the project:	
    Start > Run > cmd	
    Navigate to the location of the folder 'Predicting Biological Response' using cd	
    type the following in command line: python PredictingBiologicalResponse.py	
