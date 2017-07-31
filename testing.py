from preprocessing import *
import argparse
import dumper
import numpy
import learning
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def drop_data():
    df = learning.getData('./redis/src/')
    df.to_csv('./redis.src')


def load_data():
    data = pd.read_csv('./redis.src', na_values=['nan', 'NaN', 'Nan', ''], keep_default_na=False)
    return data


def baseline():
    df = load_data()

    categorial = ['tag', 'tag_next', 'tag_prev', 'parent_1']
    encoder = LabelEncoder()
    encoder.fit(numpy.unique(df[categorial].values))

    for name in categorial:
        df[name] = encoder.transform(df[name])

    X, testX, y, testy = train_test_split(df.drop(['newlines', 'spaces', 'tabs', 'rep'], axis=1), df['newlines'],
                                          random_state=241, test_size=0.3)

    clf = GradientBoostingClassifier(random_state=241, n_estimators=1000, learning_rate=0.1, verbose=2)
    clf.fit(X, y)
    predictions = clf.predict(testX)
    return metrics.mean_squared_error(testy, predictions)


def baseline():
    df = load_data()

    categorial = ['tag', 'tag_next', 'tag_prev', 'parent_1']
    encoder = LabelEncoder()
    encoder.fit(numpy.unique(df[categorial].values))

    for name in categorial:
        df[name] = encoder.transform(df[name])

    X, testX, y, testy = train_test_split(df.drop(['newlines', 'spaces', 'tabs', 'rep'], axis=1), df['newlines'],
                                          random_state=241, test_size=0.3)

    clf = GradientBoostingClassifier(random_state=241, n_estimators=1000, learning_rate=0.1, verbose=2)
    clf.fit(X, y)
    predictions = clf.predict(testX)
    return metrics.mean_squared_error(testy, predictions)


def baseline():
    df = load_data()

    categorial = ['tag', 'tag_next', 'tag_prev', 'parent_1']
    encoder = LabelEncoder()
    encoder.fit(numpy.unique(df[categorial].values))

    for name in categorial:
        df[name] = encoder.transform(df[name])

    X, testX, y, testy = train_test_split(df.drop(['newlines', 'spaces', 'tabs', 'rep'], axis=1), df['newlines'],
                                          random_state=241, test_size=0.3)

    clf = GradientBoostingClassifier(random_state=241, n_estimators=1000, learning_rate=0.1, verbose=2)
    clf.fit(X, y)
    predictions = clf.predict(testX)
    return metrics.mean_squared_error(testy, predictions)


def current():
    df = load_data()

    categorial = ['tag', 'tag_next', 'tag_prev', 'parent_1']
    encoder = LabelEncoder()
    encoder.fit(numpy.unique(df[categorial].values))

    for name in categorial:
        df[name] = encoder.transform(df[name])

    X, testX, y, testy = train_test_split(df.drop(['newlines', 'spaces', 'tabs', 'rep'], axis=1), df['newlines'],
                                          random_state=241, test_size=0.3)

    clf = GradientBoostingClassifier(random_state=241, n_estimators=1000, learning_rate=0.1, verbose=2, max_depth=4,
                                     min_samples_split=6, max_features='sqrt')
    clf.fit(X, y)
    predictions = clf.predict(testX)
    return metrics.mean_squared_error(testy, predictions)


def main():
    print("baseline", baseline())
    print("current", current())
    return

    df = load_data()
    categorial = ['tag', 'tag_next', 'tag_prev', 'parent_1']
    encoder = LabelEncoder()
    encoder.fit(numpy.unique(df[categorial].values))

    for name in categorial:
        df[name] = encoder.transform(df[name])

    dfX = df.drop(['newlines', 'spaces', 'tabs', 'rep'], axis=1)
    dfy = df['newlines']

    X, testX, y, testy = train_test_split(dfX, dfy, random_state=241, test_size=0.3)

    clf = GradientBoostingClassifier(random_state=241)
    cv = GridSearchCV(clf, {'learning_rate': [0.1],
                            'n_estimators': [1000],
                            'max_depth': [4],
                            'min_samples_split': [6],
                            'max_features': ['sqrt'],
                            },
                      n_jobs=-1, verbose=4, scoring='neg_mean_squared_error')
    cv.fit(dfX, dfy)

    print("cv.cv_results_", cv.cv_results_)
    print("cv.best_estimator_", cv.best_estimator_)
    print("cv.best_params_", cv.best_params_)
    print("cv.best_score_", cv.best_score_)

    clf = GradientBoostingClassifier(random_state=241, **cv.best_params_)
    clf.fit(dfX, dfy)

    #feat_imp = pd.Series(clf.feature_importances_, dfX.columns).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    #plt.show()






if __name__ == '__main__':
    main()