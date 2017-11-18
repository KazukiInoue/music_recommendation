import numpy as np
import numpy.matlib
import os
from sklearn.externals import joblib
import csv


# 配列の標準化と正規化を行う
def normalize_standardization_array(in_array, feature_name):

    mean_std_max_min = np.load('./output/'+feature_name+'_mean_std_max_min.npy')

    if len(in_array.shape) == 2:

        mean_2d = np.matlib.repmat(mean_std_max_min[0, :], in_array.shape[0], 1)
        std_2d = np.matlib.repmat(mean_std_max_min[1, :], in_array.shape[0], 1)
        max_2d = np.matlib.repmat(mean_std_max_min[2, :], in_array.shape[0], 1)
        min_2d = np.matlib.repmat(mean_std_max_min[3, :], in_array.shape[0], 1)

        # 標準化
        standardized_arr = (in_array - mean_2d)/std_2d

        # 正規化
        standed_normed_arr = (standardized_arr - max_2d) / (max_2d - min_2d)

    elif len(in_array.shape) == 3:

        mean_2d = np.matlib.repmat(mean_std_max_min[0, :], in_array.shape[1], 1)
        std_2d = np.matlib.repmat(mean_std_max_min[1, :], in_array.shape[1], 1)
        max_2d = np.matlib.repmat(mean_std_max_min[2, :], in_array.shape[1], 1)
        min_2d = np.matlib.repmat(mean_std_max_min[3, :], in_array.shape[1], 1)

        # 標準化
        standardized_arr = (in_array - mean_2d)/std_2d
        # 正規化
        standed_normed_arr = (standardized_arr - max_2d) / (max_2d - min_2d)

    else:
        raise()

    return standed_normed_arr


def main():

    # 動画ファイルをインポート(2次元)
    # fc7_dir = './data/recommendation_test/fc7_features/test/'
    video_feat_dir = './data/recommendation_test/4608hsv_features/'
    for video_feat_file in os.listdir(video_feat_dir):

        video_features = np.load(video_feat_dir + video_feat_file)
        video_features = np.array(video_features)

        # video_features = video_features.transpose()
        time_len = video_features.shape[0]

        # 音楽ファイルを動画ファイルの長さに合わせてインポート(3次元)
        aco_dir = './data/recommendation_test/aco_features/'
        aco_features = np.array([])
        for aco_index, aco_file in enumerate(os.listdir(aco_dir)):
            tmp_aco_features = np.load(aco_dir + aco_file)
            tmp_aco_features = tmp_aco_features[60 - time_len:60, :]  # 60は120bpmとしたときに30秒～50秒の曲を取得するため、本質的な意味はない
            tmp_aco_features = np.array([tmp_aco_features])

            if aco_index == 0:
                aco_features = np.array(tmp_aco_features)
            else:
                aco_features = np.concatenate([aco_features, tmp_aco_features], axis=0)

        # 正規化をかける
        # video_feat_norm = normalize_standardization_array(video_features, 'hsv')
        # aco_feat_norm = normalize_standardization_array(aco_features, 'aco')

        video_feat_norm = video_features
        aco_feat_norm = aco_features

        # 学習したMLPモデルを用いて音響特徴量を推定
        model_dir = './output/mlp_4608hsv_50_30.pkl'
        mlp = joblib.load(model_dir)
        est_aco_features = mlp.predict(video_feat_norm)

        # 距離を計算
        distance = np.array([])
        music_num = aco_feat_norm.shape[0]

        for data_iter in range(music_num):
            now_music = aco_feat_norm[data_iter, :, :]
            subtraction = now_music - est_aco_features
            square = np.power(subtraction, 2)
            tmp_distance = np.array([np.sum(square)])

            if data_iter == 0:
                distance = tmp_distance
            else:
                distance = np.concatenate([distance, tmp_distance])

        # 距離が小さい順にソート
        distance_ranking = np.argsort(distance)
        print(video_feat_file)
        print(distance_ranking[:5])
        test_data = np.array(os.listdir(aco_dir))

        # csv保存
        tmp_csv_name = video_feat_file.split('.npy')
        csv_name = './output/recommendation_result/4608hsv/4608hsv_'+tmp_csv_name[0]+'.csv'
        with open(csv_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(list(test_data[distance_ranking]))


if __name__ == '__main__':
    main()