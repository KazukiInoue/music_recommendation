import numpy as np
import numpy.matlib
import os
import time

from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib


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
def standardize_normalize_array(in_array, feature_name):

    mean_std_max_min = np.load('./output/' + feature_name + '_mean_std_max_min.npy')

    if len(in_array.shape) == 2:

        mean_2d = np.matlib.repmat(mean_std_max_min[0, :], in_array.shape[0], 1)
        std_2d = np.matlib.repmat(mean_std_max_min[1, :], in_array.shape[0], 1)
        max_2d = np.matlib.repmat(mean_std_max_min[2, :], in_array.shape[0], 1)
        min_2d = np.matlib.repmat(mean_std_max_min[3, :], in_array.shape[0], 1)

        # 標準化
        standardized_arr = (in_array - mean_2d)/std_2d

        # 正規化
        standed_normed_arr = (standardized_arr - max_2d) / (max_2d - min_2d)
    else:
        raise()

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


def main():

    # 使用する動画の特徴量の指定
    video_feat_type = '4608hsv'

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
    # video_feat_norm = standardize_normalize_array(video_features, video_feat_type)
    # aco_feat_norm = standardize_normalize_array(aco_features, 'aco')

    video_feat_norm = video_features
    aco_feat_norm = aco_features

    # フィッティング
    print('start to fit')
    mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(50,  30), activation='relu',
                       max_iter=200, random_state=1)
    begin = time.time()
    mlp.fit(video_feat_norm, aco_feat_norm)
    end = time.time()
    # assert_greater(mlp.score(X, y), 0.9)

    joblib.dump(mlp, './output/mlp_4608hsv_50_30.pkl')
    print('time: ', end-begin, 'sec')


if __name__ == '__main__':
    main()