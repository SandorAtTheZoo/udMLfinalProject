import pandas as pd

from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score

from feature_format2 import featureFormat
from toolsFeatureSelection import calcKMeans
from toolsValidation import filterScoreResults

def runClassifier(estimator, param_grid, clfData, scorers, refit_score='f1_score'):
    if len(param_grid) == 0:
        clf = estimator
        results=None
        y_pred = []
    else:
        clf, y_pred, results = grid_search_wrapper(clfData, estimator, param_grid, scorers, refit_score)

    return clf, y_pred, results

# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
def grid_search_wrapper(clfData, estimator, param_grid, scorers, refit_score):
    '''
    optimizes between different scoring methods
    :param refit_score:
    :return:
    '''
    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(estimator, param_grid, scoring=scorers, refit=refit_score,
                               cv=skf, return_train_score=True, n_jobs=1)
    grid_search.fit(clfData['xtrain'], clfData['ytrain'])

    y_pred = grid_search.predict(clfData['xtest'])

    results = pd.DataFrame(grid_search.cv_results_)
    return grid_search, y_pred, results

def tuneClassifierFromKMeans(my_dataset, features_list, classifierType, param_grid,
                             k_features, test_size, scoring='f1_score'):
    cleanedData = featureFormat(my_dataset, features_list, removePOI=False)
    mmScaler = MinMaxScaler(feature_range=(0, 1000))
    resultsScaled = mmScaler.fit_transform(cleanedData)
    fit, Xnew, featureScores = calcKMeans(resultsScaled, features_list, k_features)
    # make new feature list for passing on to tester.py for project
    dfNewFeatureList = featureScores.nlargest(k_features, 'Score')['Feature']
    print "new feature list : ", dfNewFeatureList
    features_list = ['poi'] + list(dfNewFeatureList)
    print "FEATURE LIST TO PASS TO TESTER.PY : ", features_list
    Xtrain, Xtest, yTrain, yTest = train_test_split(Xnew, cleanedData[:, 0],
                                                    test_size=test_size, random_state=42,
                                                    shuffle=True,
                                                    stratify=cleanedData[:, 0])

    # how to optimize tuning
    # https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

    clf = classifierType
    # maximize f1 to balance between precision and recall
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score)
    }
    clfData = {
        'xtrain': Xtrain,
        'ytrain': yTrain,
        'xtest': Xtest,
        'ytest': yTest
    }
    clf, y_pred, results = runClassifier(clf, param_grid, clfData, scorers,
                                         refit_score=scoring)
    if results is not None:
        print "F1 RESULTS : ", filterScoreResults(results, filterScoreType='mean_test_f1_score')
        print "Precision RESULTS : ", filterScoreResults(results,
                                                    filterScoreType='mean_test_precision_score')
        print "Recall RESULTS : ", filterScoreResults(results,
                                                      filterScoreType='mean_test_recall_score')
    return clf, features_list

def runRandomForestWithKMeans(my_dataset, features_list):
    classifierType = RandomForestClassifier()
    k_features = 5
    test_size=0.1
    param_grid = {
        'min_samples_split': [3, 5, 10],
        'n_estimators': [50, 100],
        'max_depth': [3, k_features / 2, k_features],
        'max_features': [3, k_features / 2, k_features]
    }
    clf, features_list = tuneClassifierFromKMeans(my_dataset, features_list, classifierType,
                                                  param_grid, k_features, test_size)
    # assign best parameters to classifier
    # https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
    clf = RandomForestClassifier(**clf.best_params_)
    return clf, features_list

def runAdaBoostWithKMeans(my_dataset, features_list):
    classifierType = AdaBoostClassifier()
    k_features = 5
    test_size=0.3
    param_grid = {
        'n_estimators': [50, 100,150,200],
        'learning_rate': [0.5, 1, 2,5,10]
    }
    clf, features_list = tuneClassifierFromKMeans(my_dataset, features_list, classifierType,
                                                  param_grid, k_features, test_size)
    # assign best parameters to classifier
    # https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
    clf = AdaBoostClassifier(**clf.best_params_)
    return clf, features_list

def runGradientBoostWithKMeans(my_dataset, features_list):
    classifierType = GradientBoostingClassifier()
    k_features = 5
    test_size=0.25
    param_grid = {
        'n_estimators': [50, 100,150,200],
        'learning_rate': [0.01, 0.1, 1, 2, 5],
        'subsample':[0.3, 0.6, 1]
    }
    clf, features_list = tuneClassifierFromKMeans(my_dataset, features_list, classifierType,
                                                  param_grid, k_features, test_size)
    # assign best parameters to classifier
    # https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
    clf = GradientBoostingClassifier(**clf.best_params_)
    return clf, features_list

def runSVCWithKMeans(my_dataset, features_list):
    classifierType = SVC()
    k_features = len(features_list)-1
    test_size=0.3
    param_grid = {
        'C': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1,1,2,4,8,16],
        'gamma': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.1, 0.5, 1, 2, 4]
    }
    clf, features_list = tuneClassifierFromKMeans(my_dataset, features_list, classifierType,
                                                  param_grid, k_features, test_size,
                                                  scoring='f1_score')
    # assign best parameters to classifier
    # https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
    clf = SVC(**clf.best_params_)
    return clf, features_list

def runGaussianNBWithKMeans(my_dataset, features_list):
    classifierType = GaussianNB()
    k_features = 2
    test_size=0.25
    param_grid = {}
    clf, features_list = tuneClassifierFromKMeans(my_dataset, features_list, classifierType,
                                                  param_grid, k_features, test_size,
                                                  scoring='recall_score')
    # assign best parameters to classifier
    # https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
    clf = GaussianNB()
    return clf, features_list