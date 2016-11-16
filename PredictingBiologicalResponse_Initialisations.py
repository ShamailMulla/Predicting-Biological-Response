import pandas as pd
from numpy import array
from sklearn import cross_validation
from sklearn.utils import resample

#Loading the Biological Responses Dataset into memory
def LoadData():
    # Load the entire dataset
    try:
        print "Loading the datasets..."
        training_data = pd.read_csv("train.csv")
        testing_data = pd.read_csv("test.csv")
        benchmark = pd.read_csv("benchmark.csv")
        benchmark = benchmark.drop(benchmark.columns[0], axis=1)
        benchmark_values = array([benchmark['PredictedProbability'].values]).ravel()
        
        print "Data loaded.\nEntire dataset has {} samples with {} features each.".format(*training_data.shape)     
        
        #Separating the data from the target labels for training the model
        data = training_data.drop('Activity', axis=1)
        print "Feature values dataset has {} samples with {} features each.".format(*data.shape) 
        target = training_data.columns[0]
        print "Target: {}".format(target) 
        target_labels = training_data[target]
        
        return data, target_labels, testing_data, benchmark_values
    
    except:
        print "Dataset could not be loaded. Is the dataset missing?"

#Splitting the data into test and training sets to be able to see the effectiveness of the model
def SplitData(data, labels):
    print "\nSplitting data into test and train sets..."
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=0.3, random_state=42)
    print "Train and Test sets created!"
    return X_train, X_test, y_train, y_test

    
#Creating samples
def ChooseSamples(data, labels):
    print "\nChoosing Samples..."
     
    data_samples, sample_labels = resample(data, labels, n_samples=500) 
    
    print len(data_samples), "Samples Chosen!"
    
    return data_samples, sample_labels