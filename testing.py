from preprocessing import *
import argparse
import dumper
import numpy
import learning
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

def drop_data():
    df = learning.getData('./redis/src/')
    df.to_csv('./redis.src')


def load_data():
    data = pd.read_csv('./redis.src', na_values=['nan', 'NaN', 'Nan', ''], keep_default_na=False)
    return data


def main():
    df = load_data()
    print(df.head())
    print("df shape", df.shape)

    categorial = ['tag', 'tag_next', 'tag_prev', 'parent_1']
    encoder = LabelEncoder()
    encoder.fit(numpy.unique(df[categorial].values))

    for name in categorial:
        df[name] = encoder.transform(df[name])


    X, testX, y, testy = train_test_split(df.drop(['newlines', 'spaces', 'tabs', 'rep'], axis=1), df['newlines'],
                                      random_state=241, test_size=0.3)

    clf = RandomForestClassifier(random_state=241)
    clf.fit(X, y)
    predictions = clf.predict(testX)
    print(metrics.mean_squared_error(testy, predictions))

    #gc = GridSearchCV(clf, )

    #df = learning.getData('./redis/src/')

    #print(df.head())






if __name__ == '__main__':
    main()