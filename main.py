from data_loader import *
from model_manager import *

_DEV_DATA = 'develop_v2'
_TEST_DATA = 'test_v2'
_FOLD_NUM = 5

RESULT_ROOT = './result'
if not os.path.exists(RESULT_ROOT):
    os.makedirs(RESULT_ROOT)


def test_dataloader():
    loader = CsvDataLoader(_DEV_DATA, TargetTypes.OS)
    x_mat, y_lst, feature_names = loader.get(DataPurpose.Validation, 0, True)


def train_model(model_type, use_feature_selection, target_type, k=_FOLD_NUM):
    '''
    k-fold cross validation 방법으로 모델을 학습
    :param model_type: 모델의 종류 (ModelTypes 중 하나)
    :param use_feature_selection: feature를 선택할지 여부
    :param target_type: 타겟의 종류 (TargetTypes 중 하나)
    :param k: 몇 개의 fold를 진행할 지 여부
    :return: 없음
    '''
    print('train model %s - %s ' % (model_type.value, target_type.value))
    tag = '%s_%s' % (model_type.value, target_type.value)
    if use_feature_selection:
        tag = '%s_FS' % tag

    result_direc = '%s/%s' % (RESULT_ROOT, tag)
    if not os.path.exists(result_direc):
        os.makedirs(result_direc)

    pf_path = '%s/performance_%s.csv' % (RESULT_ROOT, tag)

    dev_loader = CsvDataLoader(_DEV_DATA, target_type)
    gs_manager = GridSearchManager(model_type, result_direc)
    use_small_param = False if use_feature_selection else True
    use_smote = True if use_feature_selection else False

    for now_fold in range(k):
        print(' %s fold start:' % now_fold)
        result_dict = gs_manager.fit(dev_loader, dev_loader, now_fold,
                                     use_feature_selection, use_small_param, use_smote)

        model = result_dict[gs_manager.M]
        if model_type == ModelTypes.CAT:
            feature_lst = dev_loader.get_feature_lst(use_feature_selection)
            fi_path = '%s/%s/importance_%s.csv' % (result_direc, model_type.value, now_fold)
            export_importance_with_pandas(model, fi_path, feature_lst)

        gs_manager.predict_and_evaluate(model, dev_loader, DataPurpose.Validation,
                                        now_fold, pf_path, use_feature_selection)


def test_model(model_type, use_feature_selection, target_type, k=_FOLD_NUM):
    '''
    학습한 모델에 대해서 test set에서의 성능과 생존분석을 진행
    :param model_type: 모델의 종류 (ModelTypes 중 하나)
    :param use_feature_selection: feature를 선택할지 여부
    :param target_type: 타겟의 종류 (TargetTypes 중 하나)
    :param k: 몇 개의 fold를 진행할 지 여부
    :return: 없음
    '''
    print('test model %s - %s ' % (model_type.value, target_type.value))
    tag = '%s_%s' % (model_type.value, target_type.value)
    if use_feature_selection:
        tag = '%s_FS' % tag

    result_direc = '%s/%s' % (RESULT_ROOT, tag)
    if not os.path.exists(result_direc):
        print('There are no result directory: %s' % result_direc)

    pf_path = '%s/performance_%s.csv' % (RESULT_ROOT, tag)

    test_loader = CsvDataLoader(_TEST_DATA, target_type)
    gs_manager = GridSearchManager(model_type, result_direc)
    curve_type = CurveTypes.TTP if target_type != TargetTypes.OS else CurveTypes.OS

    for now_fold in range(k):
        print(' %s fold start:' % now_fold)

        model_path = '%s/model_%s.pkl' % (result_direc, now_fold)
        if not os.path.exists(model_path):
            continue
        model = load_obj(model_path)

        y_pred, metric = gs_manager.predict_and_evaluate(model, test_loader, DataPurpose.Test,
                                                         now_fold, pf_path, use_feature_selection)

        time_arr, event_arr = test_loader.get_kaplanmeier_info(DataPurpose.Test, curve_type, now_fold)
        sa_dict = run_survival_analysis(time_arr, event_arr, y_pred)
        df = pd.DataFrame(sa_dict)
        output_path = '%s/%s_%s.csv' % (result_direc, curve_type.value, now_fold)
        df.to_csv(output_path, index=False)


if __name__ == '__main__':
    print('HCC project')
    test_dataloader()

    # 전체 feature를 이용한 모델 학습 (feature selection)
    train_model(ModelTypes.CAT, False, TargetTypes.BR)

    # feature selection 이후
    # Logistic regression model 학습&테스트
    train_model(ModelTypes.LR, True, TargetTypes.BR)
    test_model(ModelTypes.LR, True, TargetTypes.BR)

    # CatBoost model 학습&테스트
    train_model(ModelTypes.CAT, True, TargetTypes.BR)
    test_model(ModelTypes.CAT, True, TargetTypes.BR)

    # TTP 기준 CatBoost model 학습&테스트
    train_model(ModelTypes.CAT, True, TargetTypes.PFS)
    test_model(ModelTypes.CAT, True, TargetTypes.PFS)

