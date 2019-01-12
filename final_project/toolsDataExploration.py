import sys
import matplotlib.pyplot
import numpy as np

from feature_format2 import featureFormat


def getMetadata(data_dict):
    #allKeys = data_dict["METTS MARK"]
    #print "METTS MARK item : ", allKeys
    #print "length of data_dict : ", len(data_dict)
    totalLen = len(data_dict)
    poiCount = 0
    featureQualityCount = {}
    for k,v in data_dict.items():
        #print "names of interest : ", k
        # print "type of deferral_payments : ", type(v['deferral_payments']), ":::::", \
        #     v['deferral_payments']
        if v['poi']==True:
            poiCount += 1
        for k_feature, val in v.items():
            # the not_string check below also works for booleans, so poi is an easy add
            if type(val) is not str:
                # add keys for count, min, max as nested dict under feature name for pandas
                if k_feature in featureQualityCount:
                    featureQualityCount[k_feature]['valid count'] += 1
                    if featureQualityCount[k_feature]['min'] > val:
                        featureQualityCount[k_feature]['min'] = val
                    if featureQualityCount[k_feature]['max'] < val:
                        featureQualityCount[k_feature]['max'] = val
                    if v['poi'] == True:
                        featureQualityCount[k_feature]['poiCount'] += 1
                else:
                    miniDict = {}
                    miniDict['valid count'] = 1
                    miniDict['min'] = 0
                    miniDict['max'] = 0
                    miniDict['poiCount'] = 0
                    if v['poi'] == True:
                        miniDict['poiCount'] = 1
                    featureQualityCount[k_feature] = miniDict
    #print "FEATURE COUNT : ", featureQualityCount
    #print "metadata ------------ ", featureQualityCount
    print "poi count : ", poiCount
    return featureQualityCount, totalLen

# now visually inspect the different features for any obvious problems/outliers
# set colors of poi points : https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category
def visualInspect(featuresList, data_dict):
    data = featureFormat(data_dict, featuresList, replace_NaN_with_median=True)
    feat1 = []
    feat2 = []
    for point in data:
        # poi is always the first feature (item 0)
        feat1.append(point[1])
        feat2.append(point[2])
        xVal = point[1]
        yVal = point[2]
        # set point color based on whether or not a poi
        pointColor = 'y'
        pointMarker = '+'
        if point[0] == True:
            pointColor = 'b'
            pointMarker = 'D'
        # if xVal > 1000000:
        #     print xVal,yVal
        matplotlib.pyplot.scatter(xVal, yVal, c=pointColor, marker=pointMarker)
    q1f1, q3f1, q1f2, q3f2 = calcOutliers(feat1, feat2)
    print "outlier limits : ", q1f1, q3f1, q1f2, q3f2
    for k,v in data_dict.items():
        if type(v[featuresList[1]]) is not str:
            if v[featuresList[1]] < q1f1 or v[featuresList[1]] > q3f1:
                print featuresList[1], "-----", k, "===", v[featuresList[1]]
        if type(v[featuresList[2]]) is not str:
            if v[featuresList[2]] < q1f2 or v[featuresList[2]] > q3f2:
                print featuresList[2], "-----", k, "===", v[featuresList[2]]

    matplotlib.pyplot.xlabel(featuresList[1])
    matplotlib.pyplot.ylabel(featuresList[2])
    matplotlib.pyplot.show()

# while looking at data, print out any outlier key info by taking first and third quartile * 1.5
def calcOutliers(feat1, feat2):
    # https://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy
    # https://en.wikipedia.org/wiki/Interquartile_range
    q1f1, q3f1 = np.percentile(feat1,[25,75])
    q1f2, q3f2 = np.percentile(feat2, [25,75])
    iqrf1 = q3f1-q1f1
    iqrf2 = q3f2-q1f2
    q1f1 = q1f1 - (iqrf1*1.5)
    q3f1 = q3f1 + (iqrf1*1.5)
    q1f2 = q1f2 - (iqrf2*1.5)
    q3f2 = q3f2 + (iqrf2*1.5)
    return q1f1, q3f1, q1f2, q3f2