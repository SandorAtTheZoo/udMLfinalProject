#!/usr/bin/python

import pickle
from tester import dump_classifier_and_data
from outlier_cleaner import pruneTenPercentOutliers

from toolsFeatureSelection import normalizeEmailMessages
from toolsFeatureSelection import calcTotalMonetaryValues
from toolsFeatureSelection import significantPoiEmailActivity
from toolsDataExploration import visualInspect
from toolsClassifiers import runRandomForestWithKBest, \
    runAdaBoostWithKBest, runGradientBoostWithKMeans, \
    runSVCWithKBest, runGaussianNBWithKBest

### Task 1: Select what features you'll use.

data_dict = {}

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary', 'bonus']
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

### Task 2: Remove outliers : train, remove x %, then re-train
# re-use udacity function featureFormat to remove NaNs, remove any data point where all values are zero

# found TOTAL key which contained sums of values, creating an outlier, remove it
data_dict.pop('TOTAL',None)

# now visualize some feature pairs to look for outliers
visualizeList = [
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
# new features :     ['poi', 'totalIncome', 'totalExpenses'],
for featureList in visualizeList:
    visualInspect(featureList, data_dict)

# # now train on a dataset, and then remove outliers from that fit, and re-train
# features_list = ['poi', 'deferred_income', 'exercised_stock_options']
# pruneTenPercentOutliers(cleanedData)


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
# also summarize income and expenses as new features
my_dataset = calcTotalMonetaryValues(my_dataset)

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


# # PCA setup
# data = featureFormat(my_dataset, features_list, sort_keys = True, replace_NaN_with_median=False)
#
# print " xtrain : ", data[:,1:]
# print " xtrain 2 : ", data[:,0]
# X_train, X_test, y_train, y_test = train_test_split(data[:,1:], data[:,0],
#                                                     test_size=0.2, random_state=42,
#                                                     shuffle=True)
#
# # next fit_transform the data with MinMax in preparation for PCA (required to normalize data prior to inputting to PCA, but can't call fit_transform on final pipeline, since SVM doesn't support that call)
#
# mmScaler = MinMaxScaler(feature_range=(0,1000))
# X_resultScaler = mmScaler.fit_transform(X_train)
# print "minmax scaler result : ", X_resultScaler
#
# # 3. run PCA on all features (X) and target (poi)
#
# estimators = [('reduce_dim', PCA()), ('clf', AdaBoostClassifier())]
# pipe = Pipeline(estimators)
# print "tessssssssssst"
# featureCount = range(3,10)
# estCount = range(1,100,10)
# learnRateCount = np.arange(0.1,2,0.25)
# minSampleSplitCount = range(2,10,1)
# # now run it through cross validation
# param_grid = {
#     'reduce_dim__n_components' : featureCount,
#     'clf__n_estimators' : estCount,
#     'clf__learning_rate' : learnRateCount
# }
# # 4. output components to GridSearchCV for optimal component downselect
# # EXPENSIVE :
# # find best classifier parameters with GridSearchCV
# search = GridSearchCV(pipe, param_grid, return_train_score=True)
# clf = search.fit(X_resultScaler, y_train)
# print "best CV parameter score : ", search.best_score_, search.best_params_
#
# predCVtest = search.predict(X_test)
# print "cv precision : ", cross_val_score(search, X_test, y_test, scoring='precision')
# print "cv recall : ", cross_val_score(search, X_test, y_test, scoring='recall')
#
# # more scoring
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# print "predictions, actual : ", predTest, y_test
# print "precision score : ", precision_score(y_test, predTest)
# print "recall score : ", recall_score(y_test, predTest)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

newFeatureList = ['fraction_from_poi', 'fraction_to_poi', 'totalIncome', 'totalExpenses','significant_poi_email_activity']
features_list.extend(newFeatureList)


####################################################Try RandomForestClassifier
#clf, features_list = runRandomForestWithKBest(my_dataset, features_list)
#####################################################Try AdaBoost
#clf, features_list = runAdaBoostWithKBest(my_dataset, features_list)
######################################################Try GradientBoosting
#clf, features_list = runGradientBoostWithKMeans(my_dataset, features_list)
######################################################Try SVC
#clf, features_list = runSVCWithKBest(my_dataset, features_list)
######################################################Try GaussianNB
clf, features_list = runGaussianNBWithKBest(my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)