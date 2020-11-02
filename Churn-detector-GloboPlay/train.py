import argparse
from os import path, mkdir, makedirs, umask
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from transform import dataclean as dc
from transform import dropNAN as dn

import joblib

DIR_NAME = path.dirname(__file__)

def load_label():
    y = pd.read_csv('dataset/user-status-after-test.csv', sep=',')
    
    return y

def load_data():
    df_features = pd.read_csv('dataset/weekly-infos-before-test.csv', sep=',')
    df_labels = pd.read_csv('dataset/user-status-after-test.csv', sep=',')

    print(df_features.isnull().sum())
    
    df_labels.loc[df_labels.status == 'assinante'] = 0
    df_labels.loc[df_labels.status == 'cancelou'] = 1

    label = df_labels.iloc[:,1].values
    y=label.astype('int')
    
    #print(y)
    return df_features, y

    

def transform2(X):
    cl= dc()
    dclean = cl.fit_transform(X)
    print('dclean: ', dclean)
    #tf = StandardScaler().fit_transform(X)
    cl2 = dn()
    dropn = cl2.fit_transform(dclean)
    #print('dropn: ', dropn)
    pipe = Pipeline(
        [
            ('dataclean',dc()),
            ('DropNAN',dn())
        ]
    )
    print("passou aqui")
    return pipe

def train(args):
    X,y = load_data()
    le = LabelEncoder()
    # transformation
    tf = transform2(X)

    print("-------------------------------------------- passou")
    print(tf)
    print("-------------------------------------------- passou")
    clf = MLPClassifier(max_iter=2000)
    X_tf = tf.fit_transform(X)
    clf.fit(X_tf,y)
    #print(X_tf.shape)
    clf.predict(X_tf)

    #save
    dump_folder=path.join(args['output_folder'],args['experiment_name'])
    print(dump_folder)
    if not path.exists(dump_folder):
        makedirs(dump_folder)

    # dump model
    filename = 'model_mlp_{}_v0.1.pkl'.format(args['model_name_tag'])
    joblib.dump(clf,filename=dump_folder+'/classifier')
    joblib.dump(clf,filename=path.join(dump_folder,filename))

    # dump normalization
    filename = 'tf_std_{}_v0.1.pkl'.format(args['model_name_tag'])
    joblib.dump(tf,filename=path.join(dump_folder,filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Globoplay classifier training 0.0.1')
    parser.add_argument('--experiment_name', required=True, type=str)
    parser.add_argument('--output_folder', default=path.join(DIR_NAME, 'models'), type=str)
    parser.add_argument('--model_name_tag', required=True, type=str)
    args = vars(parser.parse_args())

    train(args)