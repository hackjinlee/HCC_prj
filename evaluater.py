import enum
import os
import numpy as np
from sklearn.metrics import roc_auc_score

_T = 'time'
_E = 'event'
_N = 'number'
_PE = 'positive`s event'
_PN = 'positive`s number'
_NE = 'negative`s event'
_NN = 'negative`s number'


class Metrics(enum.Enum):
    TP = 'True Positive'
    TN = 'True Negative'
    FP = 'False Positive'
    FN = 'False Negative'

    ACC = 'Accuracy'
    SEN = 'Sensitivity'
    SPE = 'Specificity'
    PPV = 'Positive predictive values'
    NPV = 'Negative predictive values'
    F1 = 'F1score'
    F2 = 'F2score'
    AUC = 'AUC'
    

def run_survival_analysis(time_arr,event_arr, pred_arr, os_bins=np.arange(0,28,0.5)):
    '''
    모델 예측결과에 따라서 생존 분석을 진행
    :param time_arr: 실제 환자의 follow up 시간 정보
    :param event_arr: 실제 환자의 event 정보 (TTP: 암진행여부, OS: 사망 여부)
    :param pred_arr: 모델의 예측 결과
    :param os_bins: 편의를 위해 모아둘 시간대 범위정보
    :return: 결과를 저장한 dictionary.
    '''
    result_dict = dict()
    result_dict[_T] = []
    result_dict[_PE] = []
    result_dict[_PN] = []
    result_dict[_NE] = []
    result_dict[_NN] = []

    for i in range(1, len(os_bins)):
        start = os_bins[i-1]
        end = os_bins[i]

        pos_num = 0
        pos_cnt = 0
        neg_num = 0
        neg_cnt = 0

        for j in range(len(time_arr)):
            pred = pred_arr[j]
            time = time_arr[j]
            event = event_arr[j]
            if end <= time and pred >= 0.5:
                pos_num += 1
            elif end <= time and pred < 0.5:
                neg_num += 1
            elif start <= time and pred >= 0.5 and event == 1:
                pos_cnt += 1
            elif start <= time and pred < 0.5 and event == 1:
                neg_cnt += 1

        result_dict[_T].append(end)
        result_dict[_PE].append(pos_cnt)
        result_dict[_PN].append(pos_num + pos_cnt)
        result_dict[_NE].append(neg_cnt)
        result_dict[_NN].append(neg_num + neg_cnt)
    return result_dict


def export_cmdict(CM_dict, output_path, tag=None):
    tp = CM_dict[Metrics.TP.value]
    tn = CM_dict[Metrics.TN.value]
    fp = CM_dict[Metrics.FP.value]
    fn = CM_dict[Metrics.FN.value]

    accuracy = CM_dict[Metrics.ACC.value]
    sensitivity = CM_dict[Metrics.SEN.value]
    specificity = CM_dict[Metrics.SPE.value]
    precision = CM_dict[Metrics.PPV.value]
    npv = CM_dict[Metrics.NPV.value]
    f1_value = CM_dict[Metrics.F1.value]
    f2_value = CM_dict[Metrics.F2.value]

    is_need_head = os.path.exists(output_path) == False
    if is_need_head:
        mode = 'w'
    else:
        mode = 'a'


    fmt = "%s,%s\n"
    head = fmt % ('tag', ','.join([t.value for t in Metrics]))

    fmt = "%s" + ",%d" * 4 + ",%0.3f" * 7
    if tag is None:
        tag = ' '
    line = fmt % (tag,
                  tp, tn, fp, fn
                  , accuracy, sensitivity, specificity, precision,
                  npv, f1_value, f2_value)

    if Metrics.AUC.value in CM_dict:
        auc_value = CM_dict[Metrics.AUC.value]
        line = '%s,%.3f\n' % (line, auc_value)
    else:
        line += '\n'

    with open(output_path, mode) as file:
        if is_need_head:
            file.write(head)
        file.write(line)


def evaluate_model(actual_arr, pred_arr):
    '''
    모델 예측결과에 대한 성능 평가 (이진분류)
    :param actual_arr: 실제 값
    :param pred_arr: 예측 값
    :return: 결과를 저장한 dictionary
    '''
    tp = tn = fp = fn = 0
    for i in range(len(actual_arr)):
        predict = pred_arr[i]
        actual = actual_arr[i]
        if predict >= 0.5 and actual >= 1:
            tp += 1
        elif predict < 0.5 and actual == 0:
            tn += 1
        elif predict >= 0.5 and actual == 0:
            fp += 1
        elif predict < 0.5 and actual >= 1:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fn + fp) if (tp + tn + fn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = sensitivity
    f1_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_value = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0

    result_dict = dict()

    result_dict[Metrics.TP.value] = tp
    result_dict[Metrics.TN.value] = tn
    result_dict[Metrics.FP.value] = fp
    result_dict[Metrics.FN.value] = fn

    result_dict[Metrics.ACC.value] = accuracy
    result_dict[Metrics.SEN.value] = sensitivity
    result_dict[Metrics.SPE.value] = specificity
    result_dict[Metrics.PPV.value] = precision
    result_dict[Metrics.NPV.value] = npv
    result_dict[Metrics.F1.value] = f1_value
    result_dict[Metrics.F2.value] = f2_value
    result_dict[Metrics.AUC.value] = roc_auc_score(actual_arr, pred_arr)
    
    return result_dict
