import argparse
from os import path, mkdir, makedirs, umask
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline

import joblib

DIR_NAME = path.dirname(__file__)

def load_data():
    df_features = pd.read_csv('dataset/weekly-infos-before-test.csv', sep=',')
    df_labels = pd.read_csv('dataset/user-status-after-test.csv', sep=',')

    df_completo = pd.concat([df_features,df_labels],axis=1)
    #print(df_completo)
    #print(df_completo.isnull().sum())

    print("total: ",df_features.loc[df_features.user == df_labels[:,0]].count())

    print(len(df_features))
    print(len(df_labels))
    #print(len(df_features))
    df_features.dropna(inplace=True)
   # print(len(df_features))
    #df_features.fillna('a',inplace=True)
   # print(df_features.isnull().sum())
    
    X = df_features.iloc[:,1:45].values
    y = df_labels.iloc[:,1].values
    
    print("----------------Labels--------------------")
    print(y)
    print("------------------------------------------")
    print()
    print("----------------Features--------------------")
    print(X)
    print("------------------------------------------")

    return X, y

def transform(X):

    tf = StandardScaler().fit(X)

    pipe = Pipeline(
        [
            ('standard_scaler', tf)
            
        ]
    )
    #return tf
    return pipe

def train(args):
    X,y = load_data()
    le = LabelEncoder()

    #print(X.isnull().sum())

    #print(X[1,40])
    
    X[:,7] = le.fit_transform(X[:,7])
    X[:,9] = le.fit_transform(X[:,9])
    X[:,10] = le.fit_transform(X[:,10])
    X[:,40] = le.fit_transform(X[:,40])

    print(X[1,:])
    # transformation
    tf = transform(X)

    clf = MLPClassifier(max_iter=2000)

    X_tf = tf.transform(X)
    clf.fit(X_tf,y)
    print(X_tf.shape)
    clf.predict(X)

    #save
    dump_folder=path.join(args['output_folder'],args['experiment_name'])
    print(dump_folder)
    if not path.exists(dump_folder):
        makedirs(dump_folder)

    # dump model
    filename = 'model_mlp_{}_v0.1.pkl'.format(args['model_name_tag'])
    #joblib.dump(clf,filename=dump_folder+'/classifier')
    joblib.dump(clf,filename=path.join(dump_folder,filename))

    # dump normalization
    filename = 'tf_std_{}_v0.1.pkl'.format(args['model_name_tag'])
    joblib.dump(tf,filename=path.join(dump_folder,filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iris classifier training 0.0.1')
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--output_folder', default=path.join(DIR_NAME, 'models'), type=str)
    parser.add_argument('--model_name_tag', required=True, type=str)
    args = vars(parser.parse_args())

    train(args)