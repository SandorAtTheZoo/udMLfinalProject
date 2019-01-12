import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from feature_format2 import featureFormat
import pandas as pd
import numpy as np

'''
This is based off of  udacity lesson 12, question 4.
input : cleanedData -> pass in the data_dict with all of the people, as we'll be using the 
to/from email message features to create new fractional features which we'll add back to that dataset

returns : data_dict with added fractional features
'''
def normalizeEmailMessages(dataDict):
    newDataDict = dataDict
    for name, features in dataDict.items():
        fractionFromPOI = computeEmailFraction(features['from_poi_to_this_person'],
                                               features['to_messages'])
        #print "fractionEmailFrom : ", fractionFromPOI
        fractionToPOI = computeEmailFraction(features['from_this_person_to_poi'],
                                             features['from_messages'])
        #print "fractionEmailTo : ", fractionToPOI
        newDataDict[name]['fraction_from_poi'] = fractionFromPOI
        newDataDict[name]['fraction_to_poi'] = fractionToPOI
    return newDataDict

def significantPoiEmailActivity(dataDict):
    for name, features in dataDict.items():
        # if ratio of from/to poi email interaction is greater than 1/8
        if features['fraction_to_poi'] > 0:
            if (features['fraction_from_poi'] + features['fraction_to_poi']) > 0.2:
                dataDict[name]['significant_poi_email_activity'] = 1
            else:
                dataDict[name]['significant_poi_email_activity'] = 0
        else:
            dataDict[name]['significant_poi_email_activity'] = 0
    return dataDict

'''
normalize the emails from poi relative to all emails received (or vice versa for to)
- this handles NaN cases, casts to float for division, and returns 0 instead of div 0 error.
'''
def computeEmailFraction(poi_messages, all_messages):
    if all_messages != 0 and (not math.isnan(float(poi_messages)) and
                              not math.isnan(float(all_messages))):
        return float(poi_messages)/float(all_messages)
    else:
        return 0


def calcTotalMonetaryValues(dataDict):
    newDataDict = dataDict
    for name, features in dataDict.items():
        newDataDict[name]['totalIncome'] = validVal(newDataDict[name]['bonus']) + \
                                           validVal(newDataDict[name]['deferred_income']) + \
                                           validVal(newDataDict[name]['exercised_stock_options']) + \
                                           validVal(newDataDict[name]['loan_advances']) + \
                                           validVal(newDataDict[name]['long_term_incentive']) + \
                                           validVal(newDataDict[name]['restricted_stock']) + \
                                           validVal(newDataDict[name]['restricted_stock_deferred']) + \
                                           validVal(newDataDict[name]['salary']) + \
                                           validVal(newDataDict[name]['total_stock_value'])
        newDataDict[name]['totalExpenses'] = validVal(newDataDict[name]['deferral_payments']) + \
                                             validVal(newDataDict[name]['director_fees']) + \
                                             validVal( newDataDict[name]['expenses']) + \
                                             validVal(newDataDict[name]['loan_advances']) + \
                                             validVal(newDataDict[name]['total_payments'])
    return newDataDict

def validVal(featureVal):
    if type(featureVal) is not str:
        return featureVal
    else:
        return 0

'''
designed to work with featureFormat call with data_dict first
i.e.
cleanedData = featureFormat(data_dict, feature_list)
calcKMeans(cleanedData, feature_list, 10)
'''
def calcKMeans(cleanedData, feature_list, k):
    X = pd.DataFrame(cleanedData[:, 1:], columns=feature_list[1:])
    y = pd.DataFrame(cleanedData[:, 0], columns=[feature_list[0]])
    # use f_classif instead of chi2 because it supports negative numbers
    fit = SelectKBest(score_func=chi2, k=k).fit(X,y)
    Xnew = fit.transform(X)
    dfScores = pd.DataFrame(fit.scores_)
    dfColumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfColumns, dfScores], axis=1)
    featureScores.columns = ['Feature', 'Score']
    return fit, Xnew, featureScores


if __name__=="__main__":
    import pickle
    from feature_format2 import featureFormat

    features_list = ['poi',
                     'salary',
                     'deferral_payments',
                     'total_payments',
                     'loan_advances',
                     'bonus',
                     'restricted_stock_deferred',
                     'deferred_income',
                     'total_stock_value',
                     'expenses',
                     'exercised_stock_options',
                     'other',
                     'long_term_incentive',
                     'restricted_stock',
                     'director_fees',
                     'to_messages',
                     'from_poi_to_this_person',
                     'from_messages',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi'
                     ]

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # found TOTAL key which contained sums of values, creating an outlier, remove it
    data_dict.pop('TOTAL', None)

    # cleanedData = featureFormat(data_dict, features_list, removePOI=False)
    # fit, Xnew, featureScores = calcKMeans(cleanedData, features_list, 10)
    # print "scores : ", featureScores.nlargest(len(featureScores), 'Score')


    # TODO : EXTRA STUFF FROM Jupyter Notebook for troubleshooting

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from feature_format2 import featureFormat

    cleanedData = featureFormat(data_dict, features_list, removePOI=False)
    mmScaler = MinMaxScaler(feature_range=(0,1000))
    resultsScaled = mmScaler.fit_transform(cleanedData)
    fit, Xnew, featureScores = calcKMeans(resultsScaled, features_list, 10)
    Xtrain, Xtest, yTrain, yTest = train_test_split(Xnew, cleanedData[:, 0],
                                                    test_size=0.25, random_state=42, shuffle=True)

    from toolsClassifiers import runClassifier
    from sklearn.svm import SVC

    print "xtrain shape : ", Xtrain.shape
    print "xtest shape : ", Xtest.shape
    print "cleanTrainshape : ", cleanedData[:, 1:].shape
    print "len of features_list : ", len(features_list)
    print features_list

    print "Xtrain : ", Xtrain[:5], Xtrain.shape
    clf = SVC()
    param_grid = {'C': [1],
                  'kernel': ['rbf']
                  }
    clf, results = runClassifier(clf, param_grid, Xtrain, yTrain, Xtest, yTest)
    print "results : ", pd.DataFrame(results)