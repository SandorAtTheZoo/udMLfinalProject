#!/usr/bin/python

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
'''
assumes that the predictions, x, and y lists are of the same length
'''
def outlierCleaner(predictions, xList, yList, trimProportion):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual values).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (xList, yList, error).
    """
    # assumes that predicitons, xList, and yList are the same length
    # create a list of xList, yList, error
    cleaned_data = []
    for (xVal, yVal, pred) in zip(xList, yList, predictions):
        # calculate mean_squared error of each entry and add entry to cleaned_data
        err = mean_squared_error(yVal, pred)
        cleaned_data.append((xVal,yVal,err))
    # now sort list based on error, ascending
    cleaned_data = sorted(cleaned_data, key=lambda clean: clean[2]) # sort by error ascending (otherwise reverse=True)
    # discard the last int(0.1*len(cleaned_data)) entries
    newCleanLength = len(cleaned_data) - int(trimProportion*len(cleaned_data))
    cleaned_data = cleaned_data[:newCleanLength]
    
    return cleaned_data

def pruneTenPercentOutliers(cleanData):
    # reshape data to 2d numpy arrays for use in the train_test_split function
    indList = np.array(cleanData)[:,1]
    depList = np.array(cleanData)[:,2]
    newIndList = np.reshape(indList, (len(indList), 1))
    newDepList = np.reshape(depList, (len(depList), 1))

    indTrain, indTest, depTrain, depTest = train_test_split(newIndList, newDepList,
                                                            test_size=0.2, random_state=42)

    # now perform fit on data with outliers
    reg = linear_model.LinearRegression()
    reg.fit(indTrain, depTrain)
    pred = reg.predict(indTest)
    # score regression fit
    print "r^2 of initial fit : ", r2_score(depTest, pred)

    # plot result
    try:
        plt.plot(newIndList, reg.predict(newIndList), color="blue")
    except NameError:
        pass
    plt.scatter(newIndList, newDepList)
    plt.show()

    # now identify and clean outliers from previous fit
    cleaned_data = []
    try:
        # use previous prediction
        cleaned_data = outlierCleaner(pred, indTrain, depTrain, 0.1)
    except NameError:
        print "problems with regression object"

    print "cleaned data : ", cleaned_data[:5]
    # take the list of points [(a1,b1,c1),(a2,b2,c2)...] and zip to a list of 3 lists [(a1,a2,...),(b1,b2...),(c1,c2...)]
    if len(cleaned_data) > 0:
        indVals, depVals, errors = zip(*cleaned_data)
        indVals = np.reshape(np.array(indVals),(len(indVals),1))
        depVals = np.reshape(np.array(depVals), (len(depVals),1))

        # now refit the data now that outliers removed from initial fit
        try:
            reg.fit(indVals, depVals)
            plt.plot(indVals, reg.predict(indVals), color="green")
            pred = reg.predict(indTest)
            print "new r^2 score after outliers removed : ", r2_score(depTest, pred)
        except NameError:
            print " something went wrong in the refit after outliers removed"
        plt.scatter(indVals, depVals)
        plt.show()

if __name__=="__main__":
    import pickle
    import sys
    sys.path.append("../tools/")
    from feature_format2 import featureFormat

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # found TOTAL key which contained sums of values, creating an outlier, remove it
    data_dict.pop('TOTAL', None)

    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    #features_list = ['poi', 'salary', 'bonus']
    #features_list = ['poi', 'expenses', 'long_term_incentive']
    features_list = ['poi','long_term_incentive','deferral_payments']
    cleanData = featureFormat(data_dict, features_list)
    pruneTenPercentOutliers(cleanData)
