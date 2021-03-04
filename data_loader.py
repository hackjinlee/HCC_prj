import enum
import pandas as pd
import os
import random
import numpy as np


class DataPurpose(enum.Enum):
    '''
    데이터 목적 - 학습, 검증, 테스트
    '''
    Train = 'train'
    Validation = 'vali'
    Test = 'test'


class TargetTypes(enum.Enum):
    '''
    타겟 종류 - 항암제 반응(PD vs non-PD), TTP 기준, OS 기준
    '''
    BR = 'Best Response'
    PFS = 'Progression free survival'
    OS = 'Overall survival'


class CurveTypes(enum.Enum):
    '''
    생존 분석 시 y축 설정
    '''
    OS = 'overall_survival'
    TTP = 'time_to_progress'


class CsvDataLoader:
    _META_PATH = './config/meta_info.csv'
    _FOLD = 'fold'
    _SF = 'feature_name'

    def __init__(self, data_code, target_type):
        meta_df = pd.read_csv(self._META_PATH, encoding='utf-8')
        data_info = meta_df.loc[meta_df.code_name == data_code]
        if data_info.shape[0] == 0:
            print('fail to find code [%s]' % data_code)
            return
        self.data_info = data_info.iloc[0, :]
        self.file_name = data_info.file_name.values[0]
        self.full_path = data_info.full_path.values[0]
        self.split_info_path = data_info.split_path.values[0]
        self.target_type = target_type

        self.df, self.feature_lst = None, None

    def make_split_info(self, fold_num, output_path):
        """
        샘플 당 0~fold_num-1 사이의 숫자를 랜덤하게 배정
        :param fold_num: 총 fold 개수
        :param output_path: fold 정보를 출력할 경로
        :return: 없음.
        """
        data_info = self.data_info
        if self.df is None:
            self.df = pd.read_csv(self.full_path)
        df = self.df

        id_idx = int(data_info.id_idx)
        trg_idx = int(data_info.target_idx)
        id_name = df.columns[id_idx]

        fold_dict = {id_name: [], self._FOLD: []}
        length = df.shape[0]
        indexes = list(range(length))
        random.shuffle(indexes)
        pos_fold_idx = random.randint(0, fold_num-1)
        neg_fold_idx = random.randint(0, fold_num-1)
        for idx in indexes:
            now_id = df.iloc[idx, id_idx]
            trg_str = df.iloc[idx, trg_idx]
            target = 1 if trg_str.lower() == 'pd' else 0

            if target == 1:
                fold_dict[id_name].append(now_id)
                fold_dict[self._FOLD].append(pos_fold_idx)
                pos_fold_idx = pos_fold_idx + 1 if pos_fold_idx + 1 < fold_num else 0
            else:
                fold_dict[id_name].append(now_id)
                fold_dict[self._FOLD].append(neg_fold_idx)
                neg_fold_idx = neg_fold_idx + 1 if neg_fold_idx + 1 < fold_num else 0

        cnt_lst = []
        for fold in range(fold_num):
            cnt = fold_dict[self._FOLD].count(fold)
            cnt_lst.append(cnt)

        fold_df = pd.DataFrame(fold_dict)
        fold_df.to_csv(output_path, index=False)
        print('fold splitting has been finished: %s' % cnt_lst)

    def get(self, data_purpose, now_fold, use_feature_selection):
        """
        데이터를 가져오는 함수
        :param data_purpose: 데이터 목적 (학습, 검증, 테스트)
        :param now_fold: 현재 fold
        :param use_feature_selection: False=전체 feature, True 특정 feature 사용
        :return: X=모델 입력, Y=타겟, F=feature name list
        """
        if self.df is None:
            self.df = pd.read_csv(self.full_path)

        df = self.df
        data_info = self.data_info
        fold_df = pd.read_csv(self.split_info_path)

        id_idx = int(data_info.id_idx)
        id_name = df.columns[id_idx]

        if data_purpose == DataPurpose.Train:
            df = pd.merge(df, fold_df, on=id_name)
            df = df.loc[df[self._FOLD] != now_fold]
        elif data_purpose == DataPurpose.Validation:
            df = pd.merge(df, fold_df, on=id_name)
            df = df.loc[df[self._FOLD] == now_fold]
        elif data_purpose == DataPurpose.Test:
            trg_idx = int(data_info.target_idx)
            drop_idx = [i for i in range(df.shape[0]) if df.iloc[i, trg_idx] != df.iloc[i, trg_idx]]
            df = df.drop(drop_idx)

        if not use_feature_selection:
            col_lst = df.columns
            exclusion_info = data_info.exclusion_idx
            exclusion_lst = [int(token) for token in exclusion_info.split('|')]
            feature_lst = []
            for i in range(len(col_lst)):
                if col_lst[i] == self._FOLD:
                    continue
                if i not in exclusion_lst:
                    feature_lst.append(col_lst[i])
        else:
            if self.target_type == TargetTypes.BR:
                selection_path = data_info.selection_path_BR
            elif self.target_type == TargetTypes.PFS:
                selection_path = data_info.selection_path_PFS
            elif self.target_type == TargetTypes.OS:
                selection_path = data_info.selection_path_OS

            fs_df = pd.read_csv(selection_path)
            if self._SF in fs_df:
                feature_lst = fs_df[self._SF].values
            else:
                tag = 'fold_%s' % now_fold
                feature_lst = fs_df[tag].values

        input_mat = []
        for f_name in feature_lst:
            if f_name != f_name:
                continue

            temp_lst = df[f_name].values

            for i in range(len(temp_lst)):
                x = temp_lst[i]
                if isinstance(x, str):
                    try:
                        temp_lst[i] = float(x)
                    except:
                        if x == '>75000':
                            temp_lst[i] = 75000
                        elif x == '<10':
                            temp_lst[i] = 10
                        else:
                            temp_lst[i] = np.nan
            input_mat.append(temp_lst)

        x_mat = np.nan_to_num(np.array(input_mat, dtype=float))
        x_mat = x_mat.transpose()
        target_lst = None

        if self.target_type == TargetTypes.BR:
            trg_idx = int(data_info.target_idx)
            target_lst = [1 if val.lower() == 'pd' else 0 for val in df.iloc[:, trg_idx].values]
        elif self.target_type == TargetTypes.PFS:
            limit = float(data_info.ttp_limit)
            evt_idx, ttp_idx = int(data_info.pd_event),  int(data_info.ttp_duration)
            evt_lst, ttp_lst = df.iloc[:, evt_idx].values, df.iloc[:, ttp_idx].values

            target_lst = [1 if evt_lst[i] == 1 and ttp_lst[i] <= limit else 0 for i in range(len(evt_lst))]
            select_lst = [i for i in range(len(evt_lst)) if evt_lst[i] == 1 or ttp_lst[i] > limit]
            if len(select_lst) != len(evt_lst):
                x_mat = x_mat[select_lst, :]
                target_lst = np.array(target_lst)[select_lst]
        elif self.target_type == TargetTypes.OS:
            limit = float(data_info.os_limit)
            evt_idx, os_idx = int(data_info.death),  int(data_info.os_duration)
            evt_lst, os_lst = df.iloc[:, evt_idx].values, df.iloc[:, os_idx].values

            target_lst = [1 if evt_lst[i] == 1 and os_lst[i] <= limit else 0 for i in range(len(evt_lst))]
            select_lst = [i for i in range(len(evt_lst)) if evt_lst[i] == 1 or os_lst[i] > limit]
            if len(select_lst) != len(evt_lst):
                x_mat = x_mat[select_lst, :]
                target_lst = np.array(target_lst)[select_lst]

        y_arr = np.array(target_lst, dtype=float)
        self.feature_lst = feature_lst
        return x_mat, y_arr, feature_lst

    def get_kaplanmeier_info(self, data_purpose, curve_type, now_fold):
        '''
        생존 분석을 하기위한 정보 전달
        :param data_purpose: 데이터 목적
        :param curve_type: TTP기준인지 OS기준인지
        :param now_fold: 현재 fold
        :return: time 정보, event 정보
        '''
        df = self.df
        data_info = self.data_info
        fold_df = pd.read_csv(self.split_info_path)

        id_idx = int(data_info.id_idx)
        id_name = df.columns[id_idx]

        if data_purpose != DataPurpose.Test:
            df = pd.merge(df, fold_df, on=id_name)
            df = df.loc[df[self._FOLD] == now_fold]
        else:
            trg_idx = int(data_info.target_idx)
            drop_idx = [i for i in range(df.shape[0]) if df.iloc[i, trg_idx] != df.iloc[i, trg_idx]]
            df = df.drop(drop_idx)

        if curve_type == CurveTypes.OS:
            evt_idx = int(data_info.death)
            time_idx = int(data_info.os_duration)
        elif curve_type == CurveTypes.TTP:
            evt_idx = int(data_info.pd_event)
            time_idx = int(data_info.ttp_duration)

        evt_arr = df.iloc[:, evt_idx].values
        ttp_arr = df.iloc[:, time_idx].values

        if self.target_type == TargetTypes.PFS or self.target_type == TargetTypes.BR:
            limit = float(data_info.ttp_limit)
            evt_idx, ttp_idx = int(data_info.pd_event), int(data_info.ttp_duration)
            evt_lst, ttp_lst = df.iloc[:, evt_idx].values, df.iloc[:, ttp_idx].values

            select_lst = [i for i in range(len(evt_lst)) if evt_lst[i] == 1 or ttp_lst[i] > limit]
            if len(select_lst) != len(evt_arr):
                evt_arr = evt_arr[select_lst]
                ttp_arr = ttp_arr[select_lst]
        elif self.target_type == TargetTypes.OS:
            limit = float(data_info.os_limit)
            evt_idx, os_idx = int(data_info.death), int(data_info.os_duration)
            evt_lst, os_lst = df.iloc[:, evt_idx].values, df.iloc[:, os_idx].values

            select_lst = [i for i in range(len(evt_lst)) if evt_lst[i] == 1 or os_lst[i] > limit]
            if len(select_lst) != len(evt_arr):
                evt_arr = evt_arr[select_lst]
                ttp_arr = ttp_arr[select_lst]

        return ttp_arr, evt_arr
