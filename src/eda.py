
from src.feature_engineering import FeatureEngineering
from src.util import *
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
from IPython.display import display

class EDA():
    def __init__(self, params, data):
        self.params = params
        self.data = data
        self.transform_data()

    def transform_data(self):
        # apply feature engineering to full dataset
        fe = FeatureEngineering(self.params)
        fe.feature_engineering(self.data.drop('churn_in_6m', axis=1), mode='fit')
        X = fe.feature_engineering(self.data.drop('churn_in_6m', axis=1), mode='transform')
        y = self.data['churn_in_6m']

        # apply one-hot encoding for industry for EDA purpose
        enc = OneHotEncoder()
        enc.fit(X[['industry']])
        X[enc.get_feature_names_out()] = enc.transform(X[['industry']]).toarray()
        X = X.drop('industry', axis=1)
        self.transformed_data = pd.concat([X, y], axis=1)

    def plot_feature_correlation(self):
        plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(self.transformed_data.corr(), vmin=-1, vmax=1, annot=True, cmap='Greens')
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

    def plot_feature_distribution(self):
        for c in self.transformed_data.columns[:-1]:
            s0 = self.transformed_data.loc[lambda x: x.churn_in_6m==0, c]
            s1 = self.transformed_data.loc[lambda x: x.churn_in_6m==1, c]
            new_plot()
            plt.hist(s0, weights=np.ones(s0.shape[0]) / s0.shape[0], bins=20, label='0', alpha=0.5)
            plt.hist(s1, weights=np.ones(s1.shape[0]) / s1.shape[0], bins=20, label='1', alpha=0.5)
            plt.legend()
            plt.grid()
            plt.title(c)

    def plot_missing_values(self):
        df = (self.data.isnull().sum() / self.data.shape[0]).reset_index()
        df.columns = ['Column Name','Percentage Missing']
        display(df)
        df.to_csv(os.path.join(self.params['output_path'], 'missing_values.csv'))
        msno.bar(self.data, figsize=(10,6), fontsize=10)
        msno.matrix(self.data, figsize=(10,6), fontsize=10)