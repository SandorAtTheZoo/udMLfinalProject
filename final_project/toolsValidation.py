
def filterScoreResults(results, filterScoreType='mean_test_precision_score'):
    results = results.sort_values(by=filterScoreType, ascending=False)
    # features not in adaboost
    # 'param_max_depth','param_max_features','param_min_samples_split',
    # features not in SVC
    # 'param_n_estimators'
    return results[['mean_test_precision_score', 'mean_test_recall_score',
                    'mean_test_accuracy_score']].round(3).head()

if __name__ == "__main__":
    import pickle
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import confusion_matrix

    from feature_format2 import featureFormat
    from toolsFeatureSelection import normalizeEmailMessages, significantPoiEmailActivity, calcTotalMonetaryValues


    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

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
    from toolsFeatureSelection import calcKBest

    ### Store to my_dataset for easy export below.
    my_dataset = normalizeEmailMessages(data_dict)

    # create another new feature which is significant_poi_email_activity (>2% from, >17% to)
    # or approximately a ratio of 1/8 from/to
    my_dataset = significantPoiEmailActivity(my_dataset)

    my_dataset = calcTotalMonetaryValues(my_dataset)
    newFeatureList = ['fraction_from_poi', 'fraction_to_poi', 'totalIncome', 'totalExpenses',
                      'significant_poi_email_activity']
    features_list.extend(newFeatureList)

    cleanedData = featureFormat(my_dataset, features_list, removePOI=False)
    mmScaler = MinMaxScaler(feature_range=(0, 1000))
    resultsScaled = mmScaler.fit_transform(cleanedData)
    k_features = len(features_list) - 1
    fit, Xnew, featureScores = calcKBest(resultsScaled, features_list, k_features)
    Xtrain, Xtest, yTrain, yTest = train_test_split(Xnew, cleanedData[:, 0],
                                                    test_size=0.1, random_state=42, shuffle=True)

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from toolsClassifiers import runClassifier
    import pandas as pd
    from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score

    # how to optimize tuning
    # https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

    classifier = RandomForestClassifier()
    param_grid = {
        'min_samples_split': [3, 5, 10],
        'n_estimators': [100, 300],
        'max_depth': [5, 13, k_features],
        'max_features': [5, 13, k_features]
    }
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }
    clfData = {
        'xtrain': Xtrain,
        'ytrain': yTrain,
        'xtest': Xtest,
        'ytest': yTest
    }
    refit_score = 'recall_score'
    clf, y_pred, results = runClassifier(classifier, param_grid, clfData, scorers, refit_score=refit_score)

    print 'best params for {}'.format(refit_score)
    print(clf.best_params_)

    # confusion matrix for test data
    daResults = pd.DataFrame(confusion_matrix(clfData['ytest'], y_pred),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg','pos'])

    resultsForDisplay = filterScoreResults(results, filterScoreType='mean_test_recall_score')
    # display confusion matrix
    print "DA_RESULTS: ", daResults
    # display detailed classifier results filtered either by precision, recall, or accuracy
    print "RESULTS : ", resultsForDisplay

    # assign best parameters to classifier?
    # https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
    newClf = RandomForestClassifier(**clf.best_params_)
    print "classifier : ", newClf

    # use different decision threshold than 0.5
    # https://stackoverflow.com/questions/19984957/scikit-predict-default-threshold


