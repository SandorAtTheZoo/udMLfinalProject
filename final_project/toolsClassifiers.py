import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix

def runClassifier(estimator, param_grid, clfData, scorers, refit_score='precision_score'):
    if len(param_grid) == 0:
        search = estimator
        results=pd.DataFrame()
    else:
        clf, y_pred, results = grid_search_wrapper(clfData, estimator, param_grid, scorers, refit_score)
    # clf = search.fit(X_train, y_train)
    # if len(param_grid) != 0:
    #     results['best_params'] = search.best_params_
    #     print "BEST PARAMS : ", results['best_params']
    # else:
    #     results['best_params'] = ["default"]
    # results['accuracyTrain'] = clf.score(X_train, y_train)
    # predTest = clf.predict(X_test)
    # results['accuracyTest'] = clf.score(X_test, y_test)
    # results['precision'] = precision_score(y_test, predTest)
    # results['recall'] = recall_score(y_test, predTest)

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