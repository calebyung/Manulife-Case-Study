from sklearn.model_selection import StratifiedKFold
from src.feature_engineering import FeatureEngineering
import lightgbm as lgbm
import optuna
from optuna import Trial, visualization

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
