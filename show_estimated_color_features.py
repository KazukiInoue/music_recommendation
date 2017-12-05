import csv
import os
import numpy as np
import numpy.matlib


from sklearn.externals import joblib


# np.array(['a','b','c'])->np.array([['abc']])
def characters_into_one_element(characters, delimiter):
    tmp = delimiter.join(characters)
    tmp = np.array([tmp])
    characters_in_one_element = tmp.reshape(1, 1)

    return characters_in_one_element


# 配列の標準化と正規化を行う
def normalize_standardization_array(in_array, feature_name):

    mean_std_max_min = np.load('./output/min_max_mean_std/shot_'+feature_name+'_62of65_mean_std_max_min.npy')

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


def for_beat_features_recommend_music():

    # 使用する特徴量の指定
    video_feat_type = '80hsv'
    aco_feat_type = '46aco'

    # 動画特徴量をインポート(2次元)
    video_feat_dir = './data/recommendation_test/npy_shot_80hsv/'
    for video_feat_file in os.listdir(video_feat_dir):

        video_features = np.load(video_feat_dir + video_feat_file)
        video_features = np.array(video_features)

        # video_features = video_features.transpose()
        time_len = video_features.shape[0]

        # 音楽ファイルを動画ファイルの長さに合わせてインポート(3次元)
        aco_dir = '../src_data/recommendation_test_features/npy_shot_46aco/'
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
        video_feat_norm = normalize_standardization_array(video_features, video_feat_type)
        aco_feat_norm = normalize_standardization_array(aco_features, aco_feat_type)

        # 学習したMLPモデルを用いて音響特徴量を推定
        model_dir = './output/mlp_maxiter=_5000sgd_adaptive_shot_80hsv_46aco_230_120_30.pkl'
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
        csv_name = './output/recommendation_result/by_shot/by_4608hsv/4608hsv_'+tmp_csv_name[0]+'.csv'
        np.savetxt(csv_name, test_data, delimiter=',')
        with open(csv_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(list(test_data[distance_ranking]))


def for_shot_features_recommend_music():

    video_feat_type = '80hsv'
    aco_feat_type = '46aco'

    # 動画特徴量をインポート(2次元)
    video_feat_dir = '../src_data/recommendation_test_features/npy_shot_'+video_feat_type+'/'

    ranking = np.array([])
    best_worst = np.array([['tested video name', 'best5', 'worst5']])

    for video_index, video_feat_file in enumerate(os.listdir(video_feat_dir)):
        print(video_feat_file, flush=True)

        video_features = np.load(video_feat_dir + video_feat_file)
        video_features = np.array(video_features)

        # 音楽ファイルを動画ファイルの長さに合わせてインポート(3次元)
        aco_root_dir = '../src_data/recommendation_test_features/npy_shot_'+aco_feat_type+'/'
        name_index = ''
        if video_index + 1 < 10:
            name_index = '0000' + str(video_index + 1)
        elif video_index + 1 < 100:
            name_index = '000' + str(video_index + 1)

        aco_feat_dir = aco_root_dir + 'cut_by_test_video' + name_index + '/'  # = aco_root_dir + 'cut_by_test_video00012/'
        aco_features = np.array([])

        for aco_index, aco_file in enumerate(os.listdir(aco_feat_dir)):  # aco_file = test_music_00018_cut_by_test_video00012.npy
            tmp_aco_features = np.load(aco_feat_dir + aco_file)

            # 極端に演奏時間が短い曲をはじく
            if video_features.shape[0] == tmp_aco_features.shape[0]:

                tmp_aco_features = tmp_aco_features.reshape(1, tmp_aco_features.shape[0], tmp_aco_features.shape[1])

                if aco_index == 0:
                    aco_features = tmp_aco_features
                else:
                    aco_features = np.concatenate([aco_features, tmp_aco_features], axis=0)

        # 正規化

        video_feat_norm = normalize_standardization_array(video_features, video_feat_type)
        aco_feat_norm = normalize_standardization_array(aco_features, aco_feat_type)


        # 学習したMLPモデルを用いて音響特徴量を推定
        activation = 'relu'
        model_dir = './output/output_model/mlp_lbfgs_adaptive_shot_80hsv_46aco_230_120_30.pkl'
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
        tmp_ranking = np.argsort(distance)
        print(tmp_ranking[:5])

        # best5, worst5の保存
        tmp_best5 = tmp_ranking[:5].astype(np.str)
        tmp_worst5 = tmp_ranking[-5:].astype(np.str)

        best5 = characters_into_one_element(tmp_best5, ' ')
        worst5 = characters_into_one_element(tmp_worst5, ' ')
        tmp_best_worst = np.concatenate([np.array([[video_feat_file]]), best5, worst5], axis=1)

        best_worst = np.concatenate([best_worst, tmp_best_worst], axis=0)

        # ランキングの保存
        tmp_ranking = tmp_ranking.reshape(len(tmp_ranking), 1)
        tmp_ranking = np.concatenate([np.array([[video_feat_file]]), tmp_ranking], axis=0)

        if video_index == 0:
            ranking = tmp_ranking
        else:
            ranking = np.concatenate([ranking, tmp_ranking], axis=1)

    ranking_name = './output/recommendation_result/by_shot/ranking_' + video_feat_type + '_' + aco_feat_type + '.csv'
    with open(ranking_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(ranking)

    best_worst_name = './output/recommendation_result/by_shot/best5 and worst 5_' + video_feat_type + '_' + aco_feat_type + '.csv'
    with open(best_worst_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(best_worst)


if __name__ == '__main__':
    for_shot_features_recommend_music()