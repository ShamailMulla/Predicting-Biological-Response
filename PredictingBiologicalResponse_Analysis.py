# Import libraries necessary for this project
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, log_loss
from sklearn.tree import DecisionTreeClassifier
import time
#from sklearn.model_selection import cross_val_score


#Some statistics regarding the train.csv dataset
def ExploreData(data, labels):
    print "\nData Analysis"
    features_count = data.shape[1]
    count_response = np.count_nonzero(labels)
    response_percentage = count_response * 100 / data.shape[0]
    minimum = np.amin(data)
    maximum = np.amax(data)
    mean = np.mean(data)
    
    features_information_file = open("Features Information.csv","w")
    
    features_information_file.write("MoleculeID, Minimum, Maximum, Mean")
    features_information_file.write("\n")
    for i in range(0,len(minimum)):
        features_information_file.write(str(i+1))
        features_information_file.write(",")
        features_information_file.write(str(minimum[i]))
        features_information_file.write(",")
        features_information_file.write(str(maximum[i]))
        features_information_file.write(",")
        features_information_file.write(str(mean[i]))
        features_information_file.write("\n")
    features_information_file.close()
    
    print "Number of features: ", format(features_count)
    print "Number of molecules that elicited a response: ", format(count_response), "(", format(response_percentage), "%)"
    print "Number of molecules that did not elicit a response: ", format(len(data)-count_response), "(", format(100-response_percentage), "%)"
    print "To see information of every feature see 'Features Information.csv'"


#Analysing performance on SVC based on different values of C parameter    
def SVCModelComplexity(X_train, X_test, y_train, y_test):
    print "\nAnalysing SVC model complexity..."
    #Varying the c parameters
    c_parameters = [0.01, 0.1, 1, 10, 100, 1000]
    #c_parameters = [0.1, 1, 10]
    
    train_f1 = np.zeros(len(c_parameters))
    test_f1 = np.zeros(len(c_parameters))
    
    for i in range(0,len(c_parameters)):
        #Set up SVC with complexity increasing from 0.01 to 1000
        clf = SVC(C=c_parameters[i])
        
        #Train the classifier on the given data and labels
        clf.fit(X_train, y_train)
        
        #Calculate training error
        train_f1[i] = f1_score(y_train, clf.predict(X_train))        
        #Calculate testing error
        test_f1[i] = f1_score(y_test, clf.predict(X_test))
    
    print "Analysis Complete!"
    return c_parameters, train_f1, test_f1
    
 
#Analysing the performance of the Decision Tree based on different values of max_depth of the tree
def DTModelComplexity(X_train, X_test, y_train, y_test):
    print "\nAnalysing DT model complexity..."
    # Varying the depth of decision trees from 1 to 20
    max_depth = np.arange(1, 20)
    
    train_f1 = np.zeros(len(max_depth))
    test_f1 = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeClassifier(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_f1[i] = f1_score(y_train, regressor.predict(X_train))
        # Find the performance on the testing set
        test_f1[i] = f1_score(y_test, regressor.predict(X_test))

    print "Analysis Complete!"
    return max_depth, train_f1, test_f1    


#Using grid search to find best parameters for SVC and Decision Tree clsasifiers    
def FindBestParameter(data, target_labels):
    
    print "\nFinding best parameters for SVC..."
    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', SVC(probability=True))
    ])
    
    scorer = make_scorer(f1_score)
    
    n_parameters = [100, 120, 130, 145, 150, 170, 190, 195, 200, 220, 230, 235, 240, 250]
    c_parameters = [0.01, 0.1, 1, 10, 100]
    #n_parameters = [235]
    #c_parameters = [10]
    
    parameters_grid = {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': n_parameters,
        'classify__C': c_parameters
        }

    start = time.time()
    grid1 = GridSearchCV(pipe, cv=3, param_grid=parameters_grid, scoring=scorer)  
    grid1.fit(data, target_labels)  
    end = time.time()
    
    SVC_time_taken = (end - start) / 60    
    
    print "Best parameter for SVC\n", grid1.best_params_ 
    
    
    print "\nFinding best parameters for DT..."
    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', DecisionTreeClassifier())
    ])    
    
    max_depth = np.arange(2,5)
    #max_depth = [4]
    #n_parameters = [145]
    parameters_grid = {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': n_parameters,
        'classify__max_depth': max_depth
        }
    
    start = time.time()
    grid2 = GridSearchCV(pipe, cv=3, param_grid=parameters_grid, scoring=scorer)
    grid2.fit(data, target_labels) 
    end = time.time()   
    
    DT_time_taken = (end - start) / 60
    
    print "Best parameter for Decision Tree\n", grid2.best_params_    
    
    return grid1, grid2, SVC_time_taken, DT_time_taken


#Training the model to make predictionsand return the score
def MakeTrainPredictions(classifier, data, target_labels, clf_name):
    print "\nMaking predictions on training data using ", format(clf_name), "..."
    start = time.time()
    classifier.fit(data, target_labels)
    
    predictions = classifier.predict(data)
    predictions_prob = classifier.predict_proba(data)
    end = time.time()
    
    time_taken = (end - start) / 60
    
    matrix = confusion_matrix(target_labels, predictions)
    
    predictions_f1_score = f1_score(target_labels, predictions)
    
    logloss = log_loss(target_labels, predictions_prob)
    
    return logloss, predictions_f1_score, matrix, time_taken


#Running the algorithm on the test set and writing the predictions to a csv file
def PredictResponses(classifier, test_data, benchmark):
    print "\nPredicting Response Probabilities on test set..."
    
    start = time.time()
    predictions = classifier.predict(test_data)    
    predictions_probability = classifier.predict_proba(test_data)
    end = time.time()
    
    time_taken = (end - start) / 60
    
    #write predictions on the Predictions.csv file
    pred_file = open("Predictions.csv","w")    
    pred_file.write("MoleculeID, Predicted Response")
    pred_file.write("\n")
    for i in range(0,len(predictions)):
        pred_file.write(str(i+1))
        pred_file.write(",")
        pred_file.write(str(predictions[i]))
        pred_file.write("\n")
    pred_file.close()
    print "Prediction Made. Check 'Predictions.csv'"
    
    #predictions_probability_1 = []
    #write prediction probabilities to the Predictions Probabilities.csv file
    pred_prob_file = open("Predictions Probabilities.csv","w")    
    pred_prob_file.write("MoleculeID, Predicted Probability")
    pred_prob_file.write("\n")
    for i in range(0,len(predictions_probability)):
        pred_prob_file.write(str(i+1))
        pred_prob_file.write(",")
        pred_prob_file.write(str(predictions_probability[i][1]))
        #predictions_probability_1.append(predictions_probability[i][1])
        pred_prob_file.write("\n")
    pred_prob_file.close()
    print "Prediction Probabilities Estimated. Check 'Predictions Probabilities.csv'"
    predictions_probab = predictions_probability[:,[1]]
    predictions_probab = predictions_probab.ravel()
    print np.shape(predictions_probab)
      
    return predictions_probab, time_taken