#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np

from feature_format2 import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from outlier_cleaner import pruneTenPercentOutliers

from toolsFeatureSelection import normalizeEmailMessages
from toolsFeatureSelection import calcTotalMonetaryValues
from toolsFeatureSelection import significantPoiEmailActivity
from toolsDataExploration import visualInspect

### Task 1: Select what features you'll use.

data_dict = {}

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# found TOTAL key which contained sums of values, creating an outlier, remove it
data_dict.pop('TOTAL',None)
data_dict = calcTotalMonetaryValues(data_dict)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary', 'bonus']
features_list = ['poi',
                 'bonus',
                 'deferral_payments',
                 'deferred_income',
                 'exercised_stock_options',
                 'expenses',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'salary',
                 'shared_receipt_with_poi',
                 'total_payments',
                 'total_stock_value',
                 'totalIncome',
                 'totalExpenses'
                 ]

### Task 2: Remove outliers : train, remove x %, then re-train
# re-use udacity function featureFormat to remove NaNs, remove any data point where all values are zero
cleanedData = featureFormat(data_dict, features_list)
print "-----------Cleaned data sample : ", cleanedData[:5]

# TODO just running an SVM against basic data to verify anything is working
from sklearn.model_selection import train_test_split
print "X...........", cleanedData[:,1:]
print "y..............", cleanedData[:,0]
Xtrain, Xtest, yTrain, yTest = train_test_split(cleanedData[:,1:], cleanedData[:,0],
                                                test_size = 0.3, random_state=42,
                                                shuffle=True)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# clf = SVC(kernel='linear', C=1)
# # clf.fit(Xtrain, yTrain)
# # pred = clf.predict(Xtest)

clf = RandomForestClassifier()
clf.fit(Xtrain, yTrain)
pred = clf.predict(Xtest)
print "TEST prediction : ", pred
accScore = accuracy_score(yTest, pred)
print "TEST RANDOM FOREST accuracy score : ", accScore
print "TEST RANDOM FOREST precision : ", precision_score(yTest, pred)
print "TEST RANDOM FOREST recall : ", recall_score(yTest, pred)
# end TODO......................






# now train on a dataset, and then remove outliers from that fit, and re-train
features_list = ['poi', 'deferred_income', 'exercised_stock_options']
pruneTenPercentOutliers(cleanedData)


### Task 3: Create new feature(s)
# like the email example in L12Q4
### Store to my_dataset for easy export below.
my_dataset = normalizeEmailMessages(data_dict)
# visualize result
visFeatures = ['poi', 'fraction_from_poi', 'fraction_to_poi']
visualInspect(visFeatures, my_dataset)

# create another new feature which is significant_poi_email_activity (>2% from, >17% to)
# or approximately a ratio of 1/8 from/to
my_dataset = significantPoiEmailActivity(my_dataset)



# now visualize some feature pairs to look for outliers
visualizeList = [
    ['poi', 'totalIncome', 'totalExpenses'],
    ['poi','salary','bonus'],
    ['poi','deferral_payments','deferred_income'],
    ['poi','total_payments','total_stock_value'],
    ['poi','restricted_stock','restricted_stock_deferred'],
    ['poi','to_messages','from_messages'],
    ['poi','long_term_incentive','expenses'],
    ['poi','deferral_payments','long_term_incentive'],
    ['poi','deferred_income','expenses'],
    ['poi','exercised_stock_options','deferred_income'],
    ['poi', 'other', 'long_term_incentive']
]
# for featureList in visualizeList:
#     visualInspect(featureList, data_dict)



# TODO : delete? this might be getting rid of NaNs too early...delete?
# print "@@@@@@@@@@@my dataset@@@@@@@@@@@@@@ : ", my_dataset
# ### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# print "`````````````````data : ", data
# TODO : end delete

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

'''
SCALING / NORMALIZATION REQUIRED
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py

- required for SVM, but DEFINITELY for PCA

'''
# https://scikit-learn.org/stable/modules/compose.html#pipeline
# https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
# 2. Set up pipeline with StandardScaler to PCA to GridSearchCV for component selection and then SVM for classification
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

features_list = ['poi',
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'shared_receipt_with_poi',
    'total_payments',
    'total_stock_value',
    'fraction_from_poi',
    'fraction_to_poi',
    'totalIncome',
    'totalExpenses',
    'significant_poi_email_activity'
    ]
# features_list = ['poi',
#                  'fraction_from_poi',
#                  'fraction_to_poi',
#                  'expenses',
#                  'long_term_incentive'
#                  ]
# features_list = ['poi',
#                  'to_messages',
#                  'from_messages']
data = featureFormat(my_dataset, features_list, sort_keys = True, replace_NaN_with_median=False)

# TODO : FROM SAMPLE OF VISUALIZATION ABOVE create training and testing dataset
print " xtrain : ", data[:,1:]
print " xtrain 2 : ", data[:,0]
X_train, X_test, y_train, y_test = train_test_split(data[:,1:], data[:,0],
                                                    test_size=0.2, random_state=42,
                                                    shuffle=True)


features = data[:,1:]
labels = data[:,0]



# next fit_transform the data with MinMax in preparation for PCA (required to normalize data prior to inputting to PCA, but can't call fit_transform on final pipeline, since SVM doesn't support that call)

# TODO just running an SVM against basic data to verify anything is working
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

mmScaler = MinMaxScaler()
X_resultScaler = mmScaler.fit_transform(X_train)
print "minmax scaler result : ", X_resultScaler
estimators = [('reduce_dim', PCA()), ('clf', AdaBoostClassifier())]
# adaboost after cross-validation : learning_rate=1.845, n_components (PCA)=7, estimators=31
# adaboost with 0.2 test after cv : learning_rate=0.1, n_components (PCA)=7, n_estimators=51
# https://scikit-learn.org/stable/modules/ensemble.html#adaboost
# estimators = [('reduce_dim', PCA()), ('clf', AdaBoostClassifier())] #estimators=100, learning_rate=1 is decent, but still falls short in testing file
# estimators = [('reduce_dim', PCA()), ('clf', GradientBoostingClassifier(n_estimators=1000, learning_rate=0.0006))]
# estimators = [('reduce_dim', PCA()), ('clf', SVC(kernel='rbf', C=1))]
pipe = Pipeline(estimators)
print "tessssssssssst"

featureCount = range(1,10)
estCount = range(1,100,10)
learnRateCount = np.arange(0.1,2,0.25)
minSampleSplitCount = range(2,10,1)
# now run it through cross validation
param_grid = {
    'reduce_dim__n_components' : featureCount,
    'clf__n_estimators' : estCount,
    'clf__learning_rate' : learnRateCount
}




# # EXPENSIVE :
# # find best classifier parameters with GridSearchCV
# search = GridSearchCV(pipe, param_grid, return_train_score=True)
# clf = search.fit(X_resultScaler, y_train)
# print "best CV parameter score : ", search.best_score_, search.best_params_





# print "best fit for random forest from cv is n_components= 13, n_estimators=31"
#
# predCVtest = search.predict(X_test)
# print "cv precision : ", cross_val_score(search, X_test, y_test, scoring='precision')
# print "cv recall : ", cross_val_score(search, X_test, y_test, scoring='recall')


#clf = pipe.fit(X_resultScaler, y_train)


# # now get PCA result details
# # explainedVariance = pipe.named_steps['reduce_dim'].explained_variance_
# # print "transformed?????????????? : ", explainedVariance
# predTrain = clf.predict(X_resultScaler)
# print "score? : ", clf.score(X_resultScaler, y_train)
# predTest = clf.predict(X_test)
# print "test score ? : ", clf.score(X_test, y_test)
# print "explained variance of PCA : ",



# # more scoring
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# print "predictions, actual : ", predTest, y_test
# print "precision score : ", precision_score(y_test, predTest)
# print "recall score : ", recall_score(y_test, predTest)


# 3. run PCA on all features (X) and target (poi)

# 4. output components to GridSearchCV for optimal component downselect

# 5. choose remaining components to feed as features into SVM for classification (plot)

# first perform feature reduction through the use of PCA


# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# # clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)






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
from sklearn.model_selection import train_test_split
from feature_format2 import featureFormat
from toolsFeatureSelection import calcKMeans

my_dataset = calcTotalMonetaryValues(my_dataset)
newFeatureList = ['fraction_from_poi', 'fraction_to_poi', 'totalIncome', 'totalExpenses','significant_poi_email_activity']
features_list.extend(newFeatureList)

cleanedData = featureFormat(my_dataset, features_list, removePOI=False)
mmScaler = MinMaxScaler(feature_range=(0,1000))
resultsScaled = mmScaler.fit_transform(cleanedData)
k_features = 5
fit, Xnew, featureScores = calcKMeans(resultsScaled, features_list, k_features)
# make new feature list for passing on to tester.py for project
dfNewFeatureList = featureScores.nlargest(k_features,'Score')['Feature']
print "new feature list : ", dfNewFeatureList
features_list = ['poi'] + list(dfNewFeatureList)
print "FEATURE LIST TO PASS TO TESTER.PY : ", features_list
Xtrain, Xtest, yTrain, yTest = train_test_split(Xnew, cleanedData[:,0],
                                                test_size=0.1, random_state=42,
                                                shuffle=True,
                                                stratify=cleanedData[:,0])

from sklearn.ensemble import AdaBoostClassifier
from toolsClassifiers import runClassifier
import pandas as pd
from sklearn.metrics import make_scorer

# how to optimize tuning
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

clf = RandomForestClassifier()
param_grid = {
    'min_samples_split':[3,5,10],
    'n_estimators':[50,100],
    'max_depth': [3, k_features/2, k_features],
    'max_features': [3, k_features/2, k_features]
}
scorers = {
    'precision_score':make_scorer(precision_score),
    'recall_score':make_scorer(recall_score),
    'accuracy_score':make_scorer(accuracy_score)
}
clfData = {
    'xtrain':Xtrain,
    'ytrain':yTrain,
    'xtest':Xtest,
    'ytest':yTest
}
clf, y_pred, results = runClassifier(clf, param_grid, clfData, scorers, refit_score='recall_score')
from toolsValidation import filterScoreResults
print "RESULTS : ", filterScoreResults(results, filterScoreType='mean_test_recall_score')

# assign best parameters to classifier?
# https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
clf = RandomForestClassifier(**clf.best_params_)
print "classifier : ", clf

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)