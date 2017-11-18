import numpy as np
import numpy.matlib
import os
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from math import sqrt
from sklearn.metrics import mean_squared_error

import time


# ファイルの読み込みと曲の区切れ目を変えす
def load_file(max_data_num, from_dir):
    out_data = np.array([])

    for iteration, file in enumerate(os.listdir(from_dir)):
        data = np.load(from_dir + file)
        if iteration < max_data_num:
            if iteration == 0:
                out_data = data
            else:
                out_data = np.concatenate([out_data, data], axis=0)
        else:
            break

    return out_data


def record_music_begin_end(max_data_num, from_dir):
    begin_end = np.array([])
    end = 0

    for iteration, file in enumerate(os.listdir(from_dir)):
        data = np.load(from_dir + '/' + file)
        if iteration < max_data_num:
            if iteration == 0:
                begin = 0
                end = data.shape[0]
                begin_end = np.array([[begin, end]])
            else:
                begin = end
                end = end + data.shape[0]
                begin_end = np.concatenate([begin_end, [[begin, end]]], axis=0)
        else:
            break

    return begin_end


# 配列の標準化と正規化を行う
def standardize_normalize_array(in_array, title):

    mean_1d = np.mean(in_array, axis=0)
    mean_2d = np.matlib.repmat(mean_1d, in_array.shape[0], 1)
    std_1d = np.std(in_array, axis=0)
    std_2d = np.matlib.repmat(std_1d, in_array.shape[0], 1)

    standardized_arr = (in_array - mean_2d) / std_2d

    max_1d = np.max(standardized_arr, axis=0)
    max_2d = np.matlib.repmat(max_1d, in_array.shape[0], 1)
    min_1d = np.min(standardized_arr, axis=0)
    min_2d = np.matlib.repmat(min_1d, in_array.shape[0], 1)

    standed_normed_arr = (standardized_arr - min_2d) / (max_2d - min_2d)

    mean_std_max_min = np.concatenate([mean_1d.reshape(1, in_array.shape[1]),
                                       std_1d.reshape(1, in_array.shape[1]),
                                       max_1d.reshape(1, in_array.shape[1]),
                                       min_1d.reshape(1, in_array.shape[1])], axis=0)

    np.save('output/'+title+'_mean_std_max_min.npy',  mean_std_max_min)

    return standed_normed_arr


# cross validation の際に全データを学習用とテスト用に振り分ける
def distribute_train_or_test(position, x, y):
    x_dist = np.array([])
    y_dist = np.array([])

    for i in range(position.shape[0]):
        data_range = np.arange(position[i][0], position[i][1])
        tmp_x_dist = x[data_range]
        tmp_y_dist = y[data_range]

        if i == 0:
            x_dist = np.array(tmp_x_dist)
            y_dist = np.array(tmp_y_dist)
        else:
            x_dist = np.concatenate([x_dist, tmp_x_dist], axis=0)
            y_dist = np.concatenate([y_dist, tmp_y_dist], axis=0)

    return x_dist, y_dist


# SVRのパラメータのチューニング
def tuning_parameter(is_test, begin_end, n_splits, x, y):

    tuned_parameters = np.array([])

    for aco_index in range(y.shape[1]):

        # n_split交差検証法
        x_train_posi, x_test_posi, y_train_posi, y_test_posi \
            = train_test_split(begin_end, begin_end, test_size=1/n_splits, random_state=0)

        # xとyをtrain用,test用に振り分ける
        x_train, y_train = distribute_train_or_test(x_train_posi, x, y[:, aco_index])
        x_test, y_test = distribute_train_or_test(x_test_posi, x, y[:, aco_index])

        # candidate_parameters = [
        #     {'C': [1e-3, 1e-2, 1e-1], 'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 1e-1]},
        #     {'C': [1e-3, 1e-2, 1e-1], 'kernel': ['linear']},
        #     {'C': [1e-3, 1e-2, 1e-1], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [1e-3, 1e-2, 1e-1]},
        #     {'C': [1e-3, 1e-2, 1e-1], 'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-2, 1e-1]},
        # ]

        candidate_parameters = [
            {'C': [1e-3, 1e-2, 1e-1, 1, 10], 'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 1e-1, 1, 10]},
        ]

        score = 'neg_mean_squared_error'
        clf = GridSearchCV(SVR(), candidate_parameters, cv=10, scoring='%s' % score)

        start_sec = time.time()

        print('start fitting')
        clf.fit(x_train, y_train)

        end_sec = time.time()

        print('aco_feature[', aco_index, '] is finished')
        print('time to fit : %f' % (end_sec-start_sec))
        print("# Tuning hyper-parameters for %s" % score)
        print()
        print("Best parameters set found on development set: %s" % clf.best_params_)
        # print()
        # # それぞれのパラメータでの試行結果の表示
        # print("Grid scores on development set: ", clf.cv_results_)
        # print()

        if is_test is False:
            tmp_tuned_para = np.array([[clf.best_params_['C'], clf.best_params_['gamma']]])
            # filename = './output/tuned_C_gamma.npy'
            #
            # if aco_index == 0:
            #     tuned_parameters = tmp_tuned_para
            #     np.save(filename, tuned_parameters)
            # else:
            #     tuned_parameters = np.load(filename)
            #     tuned_parameters = np.concatenate([tuned_parameters, tmp_tuned_para], axis=0)
            #     np.save(filename, tuned_parameters)

        # if is_test is True:
        #     tmp_tuned_para = np.array(list(clf.best_params_.items()))
        #     filename = './output/tuned_model.npy'
        #
        #     if aco_index == 0:
        #         tuned_parameters = tmp_tuned_para
        #         np.save(filename, tuned_parameters)
        #     else:
        #         tuned_parameters = np.load(filename)
        #         tuned_parameters = np.concatenate([tuned_parameters, tmp_tuned_para], axis=0)
        #         np.save(filename, tuned_parameters)

    return tuned_parameters


# svr回帰に使用するデータ数を増やして、過学習がされていないことを確かめる
def validate_influence_of_data_num(begin_end, n_splits, tuned_parameters, x, y):

    kf = KFold(n_splits=n_splits)

    for aco_index in range(y.shape[1]):

        # tuned_c = tuned_parameters[aco_index, 0]
        # tuned_gamma = tuned_parameters[aco_index, 1]

        tuned_c = 1e3
        tuned_gamma = 1e-3

        svr_rbf = SVR(kernel='rbf', C=tuned_c, gamma=tuned_gamma)

        for data_i in range(int(begin_end.shape[0] / n_splits)):

            data_num = (data_i + 1) * 10
            small_begin_end = begin_end[:data_num, :]

            # n_split交差検証法
            cv_iteration = 0
            for music_train_index, music_test_index, in kf.split(small_begin_end):

                start = time.time()

                train_posi = small_begin_end[music_train_index]
                test_posi = small_begin_end[music_test_index]

                # xとyをtrain用,test用に振り分ける
                x_train, y_train = distribute_train_or_test(train_posi, x, y[:, aco_index])
                x_test, y_test = distribute_train_or_test(test_posi, x, y[:, aco_index])

                svr_rbf.fit(x_train, y_train)

                end = time.time()

                result = svr_rbf.predict(x_test)

                # RMSE
                rbf_rmse = sqrt(mean_squared_error(y_test, result))
                # 決定係数R^2
                scores = svr_rbf.score(x_test, y_test)
                print("used_mv_num including train and test: %f \t time: %f \t RMSE: %f \t R^2: %f"
                      % (data_num, end-start, rbf_rmse, scores))

                # cvの結果のバイナリ用の配列
                tmp_out_result = np.array([[data_num, end - start, rbf_rmse, scores]])
                if data_i == 0 and cv_iteration == 0:
                    out_result = np.array(tmp_out_result)
                else:
                    out_result = np.concatenate([out_result, tmp_out_result], axis=0)

                # 全てのデータを使った時のcvの結果を保存するための配列
                if data_i == 12:
                    tmp_out_use_all = np.array([[rbf_rmse]])
                    if cv_iteration == 0:
                        out_use_all = np.array(tmp_out_use_all)
                    else:
                        out_use_all = np.concatenate([out_use_all, tmp_out_use_all], axis=0)

                cv_iteration += 1

        # cvの結果のバイナリ保存
        # out_result = [使用したデータ数、フィットの学習に要した時間、真値とのRMSE]
        cv_filename = './output/output_cv/cv_rmse_aco_feature[' + str(aco_index) + ']'
        np.save(cv_filename, out_result)

    # 全てのデータを使った時のcvの結果を保存
    final_rmse_filename = './output/output_cv/cv_final_rmse'
    np.save(final_rmse_filename, out_use_all)


def main():

    # 使用する動画の特徴量の指定
    video_feat_type = 'hsv'

    # 交差検証法でデータをいくつに分割するかを決定
    n_splits = int(1 / 0.1)

    is_test = False

    if is_test:
        dir_x = './data/test/test_fc7_features/'
        dir_y = './data/test/test_aco_features/'
    else:
        # dir_x = './data/fc7_features/after'
        dir_x = './data/' + video_feat_type + '_features/'
        dir_y = './data/aco_features/'

    # 取得する曲数を調整を取得
    music_num = len(os.listdir(dir_x))
    max_data_num = music_num - (music_num % n_splits)

    video_features = load_file(max_data_num=max_data_num, from_dir=dir_x)
    aco_features = load_file(max_data_num=max_data_num, from_dir=dir_y)
    begin_end = record_music_begin_end(max_data_num=max_data_num, from_dir=dir_x)

    # 正規化
    # video_norm = normalize_array(video_features, 'fc7')
    video_feat_norm = standardize_normalize_array(video_features, video_feat_type)
    aco_feat_norm = standardize_normalize_array(aco_features, 'aco')

    # グリッドサーチを用いてC,gammaを決定
    # tuned_parameters = tuning_parameter(is_test, begin_end, n_splits, x, y)

    # svr回帰に使用するデータ数を増やして、過学習がされていないことを確かめる
    # tuned_parameters = np.array([1e3, 1e-3])
    # validate_influence_of_data_num(begin_end, n_splits, tuned_parameters, video_feat_norm, aco_feat_norm)

    # フィッティング
    aco_feat_dimension = aco_feat_norm.shape[1]

    if is_test is False:

        svr_rbf = SVR(kernel='rbf', C=1000, gamma=1e-3)

        for index_y in range(aco_feat_dimension):

            print('start to fit')

            start_sec = time.time()
            svr_rbf.fit(video_feat_norm, aco_feat_norm[:, index_y])
            finish_sec = time.time()

            print('Fitting takes', finish_sec-start_sec, 'second.')

            # モデルの保存
            if index_y + 1 < 10:
                tmp_filename = '00' + str(index_y+1)
            else:
                tmp_filename = '0' + str(index_y+1)

            # SVRモデルを保存
            filename = './output/output_model/' + video_feat_type + '/rbf/svr_rbf_model_'\
                       + video_feat_type + '_' + tmp_filename + '.pkl'

            joblib.dump(svr_rbf, filename)

            print(index_y, 'is finished.')


    #
    # # in_x = fc7_norm[:, -1].T
    # # print('in_x.shape = ', in_x.shape)
    # y_rbf = svr_rbf.predict(x_test)
    #
    # # 相関係数計算 # todo 分散を計算できるようにする
    # # rbf_corr = np.corrcoef(y_test, y_rbf)[0, 1]
    #
    # # RMSEを計算
    # rbf_rmse = sqrt(mean_squared_error(y_test, y_rbf))
    #
    # # print("RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))
    # print("RBF: RMSE %f" % rbf_rmse)


if __name__ == '__main__':
    main()