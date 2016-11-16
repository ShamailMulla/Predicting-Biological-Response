import PredictingBiologicalResponse_Analysis as analyse
import PredictingBiologicalResponse_Initialisations as initialise
import PredictingBiologicalResponse_Visualisations as visualise
from numpy import arange, random

def main():
    """************************************** DATA ANALYSIS **************************************"""
    #Load the Responses Data
    data, target_labels, testing_data, benchmark = initialise.LoadData()
    
    #Find out some basic statistics about the data
    analyse.ExploreData(data, target_labels)
    
    
    """**************************************** SAMPLING ****************************************"""
    #Creating samples
    data_samples, sample_labels = initialise.ChooseSamples(data, target_labels)
    #Visualising sample data distribution    
    visualise.VisualiseSamples(data_samples)
    
    
    """************************************* MODEL CREATION *************************************"""
    #Creating testing and training sets of the data in train.csv
    X_train, X_test, y_train, y_test = initialise.SplitData(data, target_labels)
    
    #Analysing complexities of SVC
    c_parameters, train_f1, test_f1 = analyse.SVCModelComplexity(X_train, X_test, y_train, y_test)
    visualise.PlotLineGraph("SVC Performance", "C value", "F1 Score", c_parameters, train_f1, test_f1, "Train Score", "Test Score")

    #Analysing complexities of Decision Tree
    max_depth, train_f1, test_f1 = analyse.DTModelComplexity(X_train, X_test, y_train, y_test)
    visualise.PlotLineGraph("Decision Tree Performance","Max Depth", "F1 Score", max_depth, train_f1, test_f1,"Train Score", "Test Score")
    
    #Getting SVC and DT classifiers having the best parameters
    classifier_svc, classifier_tree, SVC_time_taken, DT_time_taken = analyse.FindBestParameter(X_train, y_train)
    
    print "Time taken to find best parameters for SVC:", format(SVC_time_taken), " minutes."
    print "Time taken to find best parameters for DT:", format(DT_time_taken), " minutes."
    
 
    """************************************ ANALYSING MODELS ************************************"""
    #Final results of optimised models  
    SVC_log_loss, SVC_f1_score, SVC_Matrix, SVC_time_taken = analyse.MakeTrainPredictions(classifier_svc, data, target_labels, "SVC")
    #SVC_CVS_scores = cross_val_score(classifier_svc, data, target_labels, cv=3)
    DT_log_loss, DT_f1_score, DT_Matrix, DT_time_taken = analyse.MakeTrainPredictions(classifier_tree, data, target_labels, "DT")
    #SVC_Tree_scores = cross_val_score(classifier_tree, data, target_labels, cv=3)
    
    print "SVC Training F1 Score: ",SVC_f1_score
    print "SVC Log Loss Score: ", SVC_log_loss
    visualise.VisualiseHeatMaps(SVC_Matrix, "SVC Confusion Matrix")
    print "Time taken to train SVC:", format(SVC_time_taken), " minutes."
    
    #print "SVC Cross Validation Scores:\n",SVC_CVS_scores
    print "Decision Tree Training F1 Score: ",DT_f1_score 
    print "Decision Tree Log Loss Score: ", DT_log_loss
    visualise.VisualiseHeatMaps(DT_Matrix, "Decision Tree Confusion Matrix")  
    print "Time taken to train DT:", format(DT_time_taken), " minutes."
    #print "Decision Tree Cross Validation Scores:\n",SVC_Tree_scores   
       
    """************************************* FINAL MODEL *************************************"""
    #Choosing the better classifier to use on test data
    if(SVC_f1_score > DT_f1_score):
        print "Using SVC"
        predictions, time_taken = analyse.PredictResponses(classifier_svc, testing_data, benchmark)
    else:    
        print "Using Decision Tree"    
        predictions, time_taken = analyse.PredictResponses(classifier_tree, testing_data, benchmark)
    
    test_data_points = arange(0, 2500)
    sample_indexes = random.choice(test_data_points, 100)
    benchmark_samples = benchmark[sample_indexes]
    prediction_samples = predictions[sample_indexes]
    
    #Seeing the model performance against the benchmark provided
    visualise.PlotLineGraph("Final Model Performance", "Data Points", "Predicted Probability", arange(0,(len(sample_indexes))), benchmark_samples, prediction_samples, 'Benchmark', 'Predictions')
    print "Time taken to make predictions: ", format(time_taken) , " minutes." 
    

if __name__ == "__main__":
    main()