from sklearn.model_selection import StratifiedKFold
from src.feature_engineering import FeatureEngineering
from src.util import new_plot
import lightgbm as lgbm
import optuna
from optuna import Trial, visualization
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class LGBMModel:

    def __init__(self, params, data):
        self.params = params
        self.data = data
        self.create_kfolds()

    def create_kfolds(self):
        X, y = self.data.drop('active_customer_index', axis=1), self.data['active_customer_index']
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params['seed'])
        folds = {}
        for i, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
            X_trn, y_trn, X_val, y_val = X.iloc[trn_idx], y.iloc[trn_idx], X.iloc[val_idx], y.iloc[val_idx]
            fe = FeatureEngineering(self.params)
            fe.feature_engineering(X_trn, 'fit')
            X_trn, X_val = fe.feature_engineering(X_trn, 'transform'), fe.feature_engineering(X_val, 'transform')
            folds[i] = {
                'X_trn': X_trn,
                'y_trn': y_trn,
                'X_val': X_val,
                'y_val': y_val,
                'fe': fe
            }
        self.folds = folds

    def train_lgbm(self, fold, lgbm_var_params):
        model = lgbm.LGBMClassifier(**self.params['lgbm_fixed_params'], **lgbm_var_params)
        model.fit(
            X = self.folds[fold]['X_trn'],
            y = self.folds[fold]['y_trn'],
            eval_set = [(self.folds[fold]['X_val'], self.folds[fold]['y_val'])],
            eval_names = ['Validation'],
            **self.params['lgbm_fit_params']
        )
        self.folds[fold]['model'] = model
        return model.best_score_['Validation'][self.params['lgbm_fit_params']['eval_metric']]

    def tain_lgbm_kfold(self, lgbm_var_params):
        return np.mean([self.train_lgbm(fold, lgbm_var_params) for fold in range(self.params['n_fold'])])

    def Objective(self, trial):
        lgbm_var_params = dict(
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True),
            num_leaves = trial.suggest_int('num_leaves', 16, 64, log=True),
            subsample = trial.suggest_float("subsample", 0.5, 1),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1),
            reg_alpha = trial.suggest_float("reg_alpha", 1e-2, 16, log=True),
            reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 16, log=True),
        )
        return self.tain_lgbm_kfold(lgbm_var_params)


    def optimize_hp(self):
        # run optimization
        study = optuna.create_study(direction="minimize", study_name='LGBM optimization')
        study.optimize(self.Objective, timeout=self.params['hp_tune_timeout'])
        
        # save results
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.trials = study.trials_dataframe()
        
        # visualise relationship between parameter and CV score
        for c in self.trials.columns:
            if c[:7]=='params_':
                new_plot()
                self.trials.plot.scatter(c, 'value')
                plt.grid()
                plt.title(c)
                plt.show()
        return
    
