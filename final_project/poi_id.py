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
'''
again, much of this was done from the jupyter notebook. The code is hidden for readability.
'''
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
'''
uncomment the below code for the visualizations.  The same code is called from jupyter.
'''
# for featureList in visualizeList:
#     visualInspect(featureList, data_dict)

### Task 3: Create new feature(s)

'''
uncomment the visualization below if desired.  commented out while building classifiers
'''
# like the email example in L12Q4
### Store to my_dataset for easy export below.
my_dataset = normalizeEmailMessages(data_dict)
# visualize result
# visFeatures = ['poi', 'fraction_from_poi', 'fraction_to_poi']
# visualInspect(visFeatures, my_dataset)

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
TODO : This task was performed in the jupyter notebook, calling functions in the 
toolsValidation.py package.
'''


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
clf, features_list = runGradientBoostWithKMeans(my_dataset, features_list)
######################################################Try SVC
#clf, features_list = runSVCWithKBest(my_dataset, features_list)
######################################################Try GaussianNB
#clf, features_list = runGaussianNBWithKBest(my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)