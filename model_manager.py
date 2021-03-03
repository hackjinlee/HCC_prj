import enum
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *

from catboost import CatBoostClassifier, Pool, EFstrType
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
import numpy as np
import os
import shap
from sklearn import preprocessing
from imblearn.over_sampling import BorderlineSMOTE

from data_loader import *
from evaluater import *
from common import *


class ModelTypes(enum.Enum):
    LR = 'LogisticRegression'
    CAT = 'CatBoost'
    LSVM = 'LinearSVM'
    RSVM = 'RBF-SVM'
    PLS = 'PartialLeastSquares'


class GridSearchManager:
    S = 'score'
    M = 'model'
    P = 'param'

    def __init__(self, model_type, output_directory):
        self.model_type = model_type
        self.output_direc = output_directory

    def fit(self, train_loader, vali_loader, now_fold, use_selected=False, use_small_param=False, use_smote=False):
        x_train, y_train, _ = train_loader.get(DataPurpose.Train, now_fold, use_selected)
        x_vali, y_vali, _ = vali_loader.get(DataPurpose.Validation, now_fold, use_selected)

        cnt = len(y_train)
        pos_cnt = sum(y_train)
        print('...input data - %s vs %s (total %s)' % (pos_cnt, cnt - pos_cnt, cnt))

        if use_smote:
            x_train, y_train = oversample_with_smote(x_train, y_train)
            cnt = len(y_train)
            pos_cnt = sum(y_train)
            print('...after SMOTE - %s vs %s (total %s)' % (pos_cnt, cnt - pos_cnt, cnt))

        result_dict = {self.S:-1, self.M:None, self.P:None}
        params = get_model_params(self.model_type, use_small_param)
        perform_apth = '%s/performance.csv' % self.output_direc

        model_idx = 0
        for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):
            model = get_binary_model(self.model_type, prms)
            if self.model_type == ModelTypes.CAT:
                model.fit(x_train, y_train, eval_set=(x_vali, y_vali), verbose=False)
            else:
                model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            metric = evaluate_model(y_train, y_train_pred)
            train_score = metric[Metrics.F1.value]

            y_vali_pred = model.predict(x_vali)
            metric = evaluate_model(y_vali, y_vali_pred)
            vali_score = metric[Metrics.F1.value]

            if result_dict[self.S] < vali_score:
                print('....train: %.3f, vali:%.3f params: %s' % (train_score, vali_score, prms))
                tag = '%s_%s' % (now_fold, model_idx)

                export_cmdict(metric, perform_apth, tag)
                result_dict[self.S] = vali_score
                result_dict[self.M] = model
                result_dict[self.P] = prms
                output_path = '%s/%s_%s.pkl' % (self.output_direc, now_fold, model_idx)
                save_obj(model, output_path)
                model_idx += 1

        output_path = '%s/model_%s.pkl' % (self.output_direc, now_fold)
        save_obj(result_dict[self.M], output_path)
        print('param: %s' % result_dict[self.P])
        print('score: %s' % result_dict[self.S])
        return result_dict

    def predict_and_evaluate(self, model, data_loader, data_purpose, now_fold, output_path=None, use_selected=False):
        x_vali, y_vali, _ = data_loader.get(data_purpose, now_fold, use_selected)

        y_pred = model.predict(x_vali)
        metirc = evaluate_model(y_vali, y_pred)
        if self.model_type == ModelTypes.LR or self.model_type == ModelTypes.CAT:
            y_proba = model.predict_proba(x_vali)[:, 1]
            vali_auc = roc_auc_score(y_vali, y_proba)
            metirc[Metrics.AUC.value] = vali_auc

        if output_path is not None:
            tag = '%s_%s_%s' % (self.model_type.value, data_purpose.value, now_fold)
            export_cmdict(metirc, output_path, tag)

        return y_pred, metirc


def plot_shap_summary(model, x_matrix, feature_names):
    shap_values = model.get_feature_importance(data=Pool(x_matrix),
                                               type=EFstrType.ShapValues)[:, :-1]

    shap.summary_plot(shap_values, x_matrix, feature_names=feature_names)


def export_feature_importance(model, output_path, tag, names):
    fi = model.get_feature_importance()
    line = '%s,%s\n' % (tag, ','.join(map(str, fi)))
    if not os.path.exists(output_path):
        head = 'tag,%s\n' % (','.join(names))
        with open(output_path, 'w') as f:
            f.write(head)

    with open(output_path, 'a') as f:
        f.write(line)


def export_importance_with_pandas(model, output_path, names):
    fi_lst = model.get_feature_importance()
    indeces = []
    for i in range(len(fi_lst)):
        if fi_lst[i] > 0:
            indeces.append(i)

    result_dict = {'name': np.array(names)[indeces], 'importance': np.array(fi_lst)[indeces]}
    df = pd.DataFrame(result_dict)
    df.to_csv(output_path, index=False)


def oversample_with_smote(x_train, y_train, iterator=10):
    sm = BorderlineSMOTE()
    x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train)
    x_train_fin = []
    y_train_fin = []
    for i in range(iterator):
        temp_x = []
        temp_y = []
        indexes = list(range(len(y_train_sm)))
        random.shuffle(indexes)
        cnt = 0
        max_cnt = len(y_train_sm) // 10
        for j in indexes:
            x = x_train_sm[j]
            y = y_train_sm[j]
            if y == i % 2:
                temp_x.append(x)
                temp_y.append(y)
            elif cnt < max_cnt:
                temp_x.append(x)
                temp_y.append(y)
                cnt += 1
        x_sm_new, y_sm_new = sm.fit_sample(temp_x, temp_y)
        x_train_fin.extend(x_sm_new)
        y_train_fin.extend(y_sm_new)
    return x_train_fin, y_train_fin


def get_binary_model(model_type, opt_param=None, pos_weight=1):
    rs = 881109
    if model_type == ModelTypes.LR:
        if opt_param is None:
            model = LogisticRegression(solver='liblinear')
        else:
            model = LogisticRegression(C=opt_param['C'], penalty=opt_param['penalty'], solver='liblinear')
    elif model_type == ModelTypes.LSVM:
        model = SVC(kernel='linear', verbose=False, C=opt_param['C'])
    elif model_type == ModelTypes.RSVM:
        model = SVC(kernel='rbf', verbose=3,
                    C=opt_param['C'], gamma=opt_param['gamma'])
    elif model_type == ModelTypes.CAT:
        if opt_param is None:
            model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=3, loss_function='Logloss',
                                       scale_pos_weight=pos_weight, use_best_model=True,
                                       early_stopping_rounds=30, thread_count=-1)
        else:
            model = CatBoostClassifier(iterations=1000,
                                       eval_metric='F1',
                                       depth=opt_param['depth'],
                                       learning_rate=opt_param['learning_rate'],
                                       l2_leaf_reg=opt_param['l2_leaf_reg'],
                                       scale_pos_weight=opt_param['scale_pos_weight'],
                                       border_count=opt_param['border_count'],
                                       colsample_bylevel=opt_param['colsample_bylevel'],
                                       loss_function='Logloss',
                                       use_best_model=True,
                                       early_stopping_rounds=100,
                                       thread_count=10
                                       )
    elif model_type == ModelTypes.PLS:
        if opt_param is None:
            model = PLSRegression(n_components=5)
        else:
            model = PLSRegression(n_components=opt_param['n_components'], max_iter=2000, tol=1e-4)
    return model


def get_model_params(model_type, use_small=False):
    if model_type == ModelTypes.LR:
        params = {
            "C": np.logspace(-3, 3, 7),
            "penalty": ["l1", "l2"]
        }
    elif model_type == ModelTypes.RSVM:
        params = {
            "C": np.logspace(-3, 3, 7),
            "gamma": np.logspace(-3, 3, 7),
        }
    elif model_type == ModelTypes.LSVM:
        params = {
            "C": np.logspace(-3, 2, 6),
        }
    elif model_type == ModelTypes.CAT and use_small:
        params = {
            'scale_pos_weight': [1.0],
            'depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.001],
            'l2_leaf_reg': [8, 16, 32],
            'border_count': [5, 50, 125],
            'colsample_bylevel': [0.05, 0.1]
        }
    elif model_type == ModelTypes.CAT:
        params = {
            'scale_pos_weight': [0.8, 1.0, 1.2],
            'depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.001, 0.05, 0.01],
            'l2_leaf_reg': [4, 16, 64, 96, 128, 160, 192, 224],
            'border_count': [5, 50, 100, 150, 200, 250],
            'colsample_bylevel': [0.1, 0.15, 0.2, 0.25]
        }
    elif model_type == ModelTypes.PLS:
        params = {
            "n_components": list(range(5, 30)),
        }
    return params