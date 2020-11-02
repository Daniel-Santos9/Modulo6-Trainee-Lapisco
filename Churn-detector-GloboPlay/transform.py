from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class dataclean(BaseEstimator, TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        print("entrou na funçãooooooo")
        features_numerical = df.drop(columns=['sexo', 'tipo_de_cobranca','cidade', 'estado'])

        df_buffer = features_numerical.loc[features_numerical['week']<18]
        mean_fn = df_buffer.groupby(['user']).agg('mean')

        print("passou aqui excluindo coisas")

        return mean_fn
    
class dropNAN(BaseEstimator, TransformerMixin):

    def fit(self, mean_fn):
        return self

    def transform(self, mean_fn):

        #df_labels = pd.read_csv('dataset/user-status-after-test.csv', sep=',')
        #df_labels.loc[df_labels.status == 'assinante'] = 0
        #df_labels.loc[df_labels.status == 'cancelou'] = 1

        #label = df_labels.iloc[:,1].values
        #print("labels", label)

        #mean_fn["label"] = label

        #print('features: ',mean_fn.shape)
        mean_fn.fillna(0, inplace=True)
        #df_clean = mean_fn.dropna(how='any', inplace=False)

        X = mean_fn.iloc[:,1:40].values
        #y = df_clean.iloc[:,40].values
        #y=y.astype('int')

        return X