import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import RobustScaler

class FeatureEngineering:

    def __init__(self, params):
        self.params = params
        
    def feature_engineering(self, X, mode):
        # remove client_id as it is not a feature
        X = X.drop('client_id', axis=1)

        # sex - one-dimension encoding
        X.sex = X.sex.fillna('Unknown').map({'M':-1, 'Unknown':0, 'F':1})

        # treate missing industry as separate category; group smaller industries into one; one-hot encoding the rest
        X.industry = X.industry.fillna(-1).astype(int)
        X.industry = np.select([X.industry.isin([10,20,11,15,17,14]), True],[0, X.industry])
        for i in [99,12,13,0]:
            X[f'industry_{i}'] = (X.industry == i).astype(int)
        X = X.drop('industry', axis=1)

        # identify invalid year_of_birth; create feature as age
        X.year_of_birth = np.select([X.year_of_birth==1, True],[np.nan, X.year_of_birth])
        X['age'] = datetime.strptime(self.params['analysis_date'], '%Y-%m-%d').year - X.year_of_birth
        X = X.drop('year_of_birth', axis=1)

        # identify invalid join_date; create feature as years of relationship
        X.join_date = X.join_date.astype(str)
        X.join_date = np.select([X.join_date.isin(['0001-01-01 00:00:00.000','00:00:00']), True],[np.nan, X.join_date])
        X.join_date = pd.to_datetime(X.join_date)
        X['yrs_of_relationship'] = (np.datetime64(self.params['analysis_date'], 'D') - X.join_date) / np.timedelta64(1, 'Y')
        X = X.drop('join_date', axis=1)

        # remove with_product_a as it has zero variance
        X = X.drop('with_product_a', axis=1)

        # one-hot encoding for product b
        X.with_product_b = X.with_product_b.map({'N':0, 'Y':1})

        # total_aum - log transform
        X.total_aum = X.total_aum.apply(lambda x: np.log(x + 1))

        # no_of_transaction - log transform
        X.no_of_transaction = X.no_of_transaction.apply(lambda x: np.log(x + 1))

        # income - log transform
        X.income = X.income.apply(lambda x: np.log(x + 1))

        # channel - one-dimension encoding
        X.channel = X.channel.map({'Agency':-1, 'Others':0, 'Broker':1})

        # preset feature order
        X = X[['sex','industry_99','industry_12','industry_13','industry_0','age','yrs_of_relationship','with_product_b','total_aum','no_of_transaction','income','channel']]

        # imputation
        if mode == 'fit':
            self.features_mean = X.median()
        X = X.fillna(self.features_mean)

        # scaling
        if mode == 'fit':
            self.scaler = RobustScaler()
            self.scaler.fit(X[['age','yrs_of_relationship','total_aum','no_of_transaction','income']])
        X[['age','yrs_of_relationship','total_aum','no_of_transaction','income']] = self.scaler.transform(X[['age','yrs_of_relationship','total_aum','no_of_transaction','income']])

        # return new X if mode is transform
        if mode == 'fit':
            return
        elif mode == 'transform':
            return X






