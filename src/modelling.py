from sklearn.model_selection import StratifiedKFold, train_test_split
from src.feature_engineering import FeatureEngineering
from src.util import new_plot
import lightgbm as lgbm
import optuna
from optuna import Trial, visualization
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, log_loss
from IPython.display import display
import os


class LGBMModel:

    def __init__(self, params, data):
        self.params = params
        self.data = data
        self.partition_data()

    def partition_data(self):
        folds = {}
        # create hold-out set as final test set
        X, X_test, y, y_test = train_test_split(self.data.drop('churn_in_6m', axis=1), 
                                                self.data['churn_in_6m'], 
                                                test_size = self.params['test_size'], 
                                                random_state = self.params['seed'],
                                                shuffle = True,
                                                stratify = self.data['churn_in_6m'])
        folds['test'] = {'X_test': X_test, 'y_test':y_test}
        # create k-folds
        X, y = pd.DataFrame(X, columns = self.data.columns[:-1]), pd.Series(y)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params['seed'])
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
        X, y = self.folds[fold]['X_trn'], self.folds[fold]['y_trn']
        oversample = SMOTE(sampling_strategy=self.params['smote_ratio'], random_state=self.params['seed'])
        X, y = oversample.fit_resample(X, y)
        model = lgbm.LGBMClassifier(**self.params['lgbm_fixed_params'], **lgbm_var_params)
        model.fit(
            X = X,
            y = y,
            categorical_feature = [1],
            eval_set = [(self.folds[fold]['X_val'], self.folds[fold]['y_val'])],
            eval_names = ['Validation'],
            **self.params['lgbm_fit_params']
        )
        self.folds[fold]['model'] = model
        return model.best_score_['Validation'][self.params['lgbm_fit_params']['eval_metric']]

    def train_lgbm_kfold(self, lgbm_var_params):
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
        return self.train_lgbm_kfold(lgbm_var_params)


    def optimize_hp(self):
        # run optimization
        study = optuna.create_study(direction="minimize", study_name='LGBM optimization')
        study.optimize(self.Objective, timeout=self.params['hp_tune_timeout'])
        # save results
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.trials = study.trials_dataframe()
        
    def visualize_hp_tune_results(self):
        for c in self.trials.columns:
            if c[:7]=='params_':
                new_plot()
                self.trials.plot.scatter(c, 'value')
                plt.grid()
                plt.title(c)
                plt.show()


    def evaluate(self, y, y_pred, y_pred_proba):
        return {'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'logloss': log_loss(y, y_pred_proba),
                'conf': confusion_matrix(y, y_pred),
                'pr_curve': precision_recall_curve(y, y_pred_proba)
        }
        
    def evaluate_trn(self):
        eval_all = []
        for fold in range(self.params['n_fold']):
            model = self.folds[fold]['model']
            y_pred_proba = model.predict_proba(self.folds[fold]['X_trn'])[:,1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            y = self.folds[fold]['y_trn']
            eval_all.append(self.evaluate(y, y_pred, y_pred_proba))
        self.eval_trn = {
            'precision': np.mean([eval['precision'] for eval in eval_all]),
            'recall': np.mean([eval['recall'] for eval in eval_all]),
            'f1': np.mean([eval['f1'] for eval in eval_all]),
            'logloss': np.mean([eval['logloss'] for eval in eval_all])
        }

    def evaluate_val(self):
        eval_all = []
        for fold in range(self.params['n_fold']):
            model = self.folds[fold]['model']
            y_pred_proba = model.predict_proba(self.folds[fold]['X_val'])[:,1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            y = self.folds[fold]['y_val']
            eval_all.append(self.evaluate(y, y_pred, y_pred_proba))
        self.eval_val = {
            'precision': np.mean([eval['precision'] for eval in eval_all]),
            'recall': np.mean([eval['recall'] for eval in eval_all]),
            'f1': np.mean([eval['f1'] for eval in eval_all]),
            'logloss': np.mean([eval['logloss'] for eval in eval_all])
        }

    def evaluate_test(self):
        y_test_pred_proba = np.zeros(self.folds['test']['y_test'].shape[0])
        for fold in range(self.params['n_fold']):
            fe = self.folds[fold]['fe']
            X_test = self.folds['test']['X_test']
            X_test = fe.feature_engineering(X_test, 'transform')
            model = self.folds[fold]['model']
            y_test_pred_proba += model.predict_proba(X_test)[:,1] / self.params['n_fold']
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        self.eval_test = self.evaluate(self.folds['test']['y_test'], y_test_pred, y_test_pred_proba)

    def evaluate_all(self):
        # evaluate train, val, test sets
        self.evaluate_trn()
        self.evaluate_val()
        self.evaluate_test()

        # report performance metrics
        self.eval_metrics = pd.DataFrame({'Train':[self.eval_trn['precision'], self.eval_trn['recall'], self.eval_trn['f1'], self.eval_trn['logloss']],
                                        'Validation':[self.eval_val['precision'], self.eval_val['recall'], self.eval_val['f1'], self.eval_val['logloss']],
                                        'Test':[self.eval_test['precision'], self.eval_test['recall'], self.eval_test['f1'], self.eval_test['logloss']]},
                                        index = ['Precision','Recall','F1','Logloss'])
        display(self.eval_metrics)
        self.eval_metrics.to_csv(os.path.join(self.params['output_path'], 'eval_metrics.csv'))

        # report test set confusion matrix
        self.test_confusion_matrix = pd.DataFrame(self.eval_test['conf'], columns=['Pred_0', 'Pred_1'], index=['True_0','True_1'])
        self.test_confusion_matrix.to_csv(os.path.join(self.params['output_path'], 'test_confusion_matrix.csv'))
        display(self.test_confusion_matrix)

        # report Precision-Recall Curve
        new_plot()
        plt.plot(self.eval_test['pr_curve'][2], self.eval_test['pr_curve'][0][:-1], label='Precision')
        plt.plot(self.eval_test['pr_curve'][2], self.eval_test['pr_curve'][1][:-1], label='Recall')
        plt.title('Precision-Recall Curve for Test Set')
        plt.grid()
        plt.legend()
        plt.show()

    def show_feature_importance(self):
        feat_imp = []
        for fold in range(self.params['n_fold']):
            model = self.folds[fold]['model']
            model.set_params(importance_type='gain')
            imp_gain = model.feature_importances_
            model.set_params(importance_type='split')
            imp_split = model.feature_importances_
            output = pd.concat([
                        pd.DataFrame({'feat':model.feature_name_, 'imp':imp_gain}).assign(imp_type='gain', imp=lambda x: x.imp.rank()),
                        pd.DataFrame({'feat':model.feature_name_, 'imp':imp_split}).assign(imp_type='split', imp=lambda x: x.imp.rank())
                    ], axis=0).assign(fold=fold)
            feat_imp.append(output)
        feat_imp = pd.concat(feat_imp)
        feat_imp_grouped = feat_imp.groupby('feat').imp.mean().sort_values(ascending=False).reset_index()
        feat_imp_grouped.to_csv(os.path.join(self.params['output_path'], 'feat_imp_grouped.csv'))
        display(feat_imp_grouped)
        feat_imp_grouped.sort_values('imp', ascending=True).set_index('feat').plot.barh(width=0.1)
        plt.title('Feature Importance (Averaged Rank)')
        self.feat_imp = feat_imp
        self.feat_imp_grouped = feat_imp_grouped

    def show_feature_correlation(self):
        fe = FeatureEngineering(self.params)
        fe.feature_engineering(self.data.drop('churn_in_6m', axis=1), mode='fit')
        X = fe.feature_engineering(self.data.drop('churn_in_6m', axis=1), mode='transform')
        y = self.data['churn_in_6m']







    
