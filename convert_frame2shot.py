import csv
import math
import numpy as np
import os
import sys


def calc_max_min_and_concatenate_features(src_array, output_feature_type):

    out_arr = np.array([])

    if output_feature_type == "mean_only":
        mean_features = np.mean(src_array, axis=0)
        out_arr = mean_features

    elif output_feature_type == "mean_and_std":
        mean_features = np.mean(src_array, axis=0)
        std_features = np.std(src_array, axis=0)
        out_arr = np.concatenate([mean_features, std_features], axis=0)

    out_arr = out_arr.reshape(1, len(out_arr))

    return out_arr


def for_test_convert_frame2shot_features(threshold):

    input_feature_type = "40aco"
    output_feature_type = "mean_and_std"

    root_shot_img_dir = 'C:/MUSIC_RECOMMENDATION/src_data/shots_OMV62of65_improved/'

    from_feat_dirs = None
    before_foot = None
    frame_length = None
    frame_hop = None
    to_root_dir = None
    after_foot = None

    if input_feature_type == "46aco":

        from_feat_dirs = "../src_data/train_features/OMV62of65_npy_frame_46aco/"

        before_foot = "_frame_46aco.npy"

        frame_length = 0.03
        frame_hop = 0.01

        if output_feature_type == "mean_only":
            to_root_dir = "../src_data/recommendation_test_features/for_test_OMV62of65_npy_shot_46aco/"
            after_foot = "_shot_46aco.npy"

        elif output_feature_type == "mean_and_std":
            to_root_dir = "../src_data/recommendation_test_features/for_test_OMV62of65_npy_shot_92aco/"
            after_foot = "_shot_92aco.npy"

    elif input_feature_type == "40aco":

        from_feat_dirs = "../src_data/train_features/OMV62of65_npy_frame_40aco/"

        before_foot = "_frame_40aco.npy"

        frame_length = 2048 / 44100
        frame_hop = 512 / 44100

        if output_feature_type == "mean_only":
            to_root_dir = "../src_data/recommendation_test_features/for_test_OMV62of65_npy_shot_40aco/"
            after_foot = "_shot_40aco.npy"

        elif output_feature_type == "mean_and_std":
            to_root_dir = "../src_data/recommendation_test_features/for_test_OMV62of65_npy_shot_80aco/"
            after_foot = "_shot_80aco.npy"

    # ショット検出されたフレームを読み込む
    for video_index, video_folder in enumerate(os.listdir(root_shot_img_dir)):

        shot_img_dir = root_shot_img_dir + video_folder

        if len(os.listdir(shot_img_dir)) < threshold:  # ショット検出できていない動画があるので、その動画は扱わない
            continue
        else:

            to_dir = to_root_dir + "cut_by_" + video_folder + "/"

            # ショット発生時間の取得
            shot_time_posis = np.array([])

            for img_file in os.listdir(shot_img_dir):
                # ファイルの名前をもとに特徴量をまとめる
                tmp_shot_posi = img_file.split('_')  # ->[test,video00005,01906,63.596867.png]
                tmp_shot_posi = tmp_shot_posi[3].split('.png')  # -> [63.59687]
                tmp_shot_posi = np.array([float(tmp_shot_posi[0])])  # 63.59687

                if shot_time_posis.size == 0:
                    shot_time_posis = tmp_shot_posi
                else:
                    shot_time_posis = np.concatenate([shot_time_posis, tmp_shot_posi])

            # フレーム音響特徴量のnpyを読み込む
            for frame_feat_file in os.listdir(from_feat_dirs):

                frame_features = np.load(from_feat_dirs + frame_feat_file)

                shot_features = np.array([])

                # ショット発生時間を用いてフレーム特徴量を分割
                now_row_index = 0
                prev_row_index = 0
                for shot_index, shot_time_posi in enumerate(shot_time_posis):

                    frame_time_posi = frame_length

                    if shot_index == 0:  # 最初のフレームに対しては計算しない
                        continue
                    else:
                        while frame_time_posi <= shot_time_posi:
                            now_row_index += 1
                            frame_time_posi = now_row_index * frame_hop + frame_length

                        if now_row_index >= frame_features.shape[0]:  # 音楽よりも動画の方が長い場合
                            break
                        else:

                            this_shot_features = frame_features[prev_row_index:now_row_index, :]
                            tmp_shot_features = calc_max_min_and_concatenate_features(this_shot_features, output_feature_type)

                            prev_row_index = now_row_index

                            if shot_features.size == 0:
                                shot_features = tmp_shot_features
                            else:
                                shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

                # 最後のショットの情報をここで得る
                # 動画よりも音楽のほうが長い場合は、最後のショットに残りすべてのフレーム音響特徴量を割り当てる
                if now_row_index < frame_features.shape[0] and shot_features.shape[0] < len(shot_time_posis):
                    this_shot_features = frame_features[now_row_index:, :]
                    tmp_shot_features = calc_max_min_and_concatenate_features(this_shot_features, output_feature_type)

                    shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

                tmp_save_name = frame_feat_file.split(before_foot)
                save_name = tmp_save_name[0]  # test_music_00123

                npy_name = to_dir + save_name + after_foot
                np.save(npy_name, shot_features)

                print(video_index + 1, '曲目が終了')
                print(shot_features.shape)


def for_training_convert_frame2shot_features(threshold):

    input_feature_type = "40aco"
    output_feature_type = "mean_only"  # mean_only or mean_and_std

    if output_feature_type != "mean_only" and output_feature_type != "mean_and_std":
        print("Error! Please specify 'mean_only' or 'mean_and_std' for output_feature_type.")
        exit(1)

    root_shot_img_dirs = ["../src_data/shots_OMV200_improved/",  # timposiとの兼ね合いがあるので,必ず0番目の要素にOMV200の情報を入れること
                          "../src_data/shots_OMV62of65_improved/"]

    from_feat_dirs = ["", ""]
    to_dirs = ["", ""]
    before_foot = ""
    after_foot = ""
    frame_length = 0
    frame_hop = 0

    if input_feature_type == "46aco":

        from_feat_dirs = ["../src_data/train_features/OMV200_npy_frame_46aco/",
                          "../src_data/train_features/OMV62of65_npy_frame_46aco/"]

        before_foot = "_frame_46aco.npy"

        frame_length = 0.03
        frame_hop = 0.01

        if output_feature_type == "mean_only":
            to_dirs = ["../src_data/train_features/OMV200_npy_shot_46aco/",
                       "../src_data/train_features/OMV62of65_npy_shot_46aco/"]

        elif output_feature_type == "mean_and_std":
            to_dirs = ["../src_data/train_features/OMV200_npy_shot_92aco/",
                       "../src_data/train_features/OMV62of65_npy_shot_92aco/"]

            after_foot = "_shot_92aco.npy"

    elif input_feature_type == "40aco":

        from_feat_dirs = ["../src_data/train_features/OMV200_npy_frame_40aco/",
                          "../src_data/train_features/OMV62of65_npy_frame_40aco/"]

        before_foot = "_frame_40aco.npy"

        frame_length = 2048 / 44100
        frame_hop = 512 / 44100

        if output_feature_type == "mean_only":
            to_dirs = ["../src_data/train_features/OMV200_npy_shot_40aco_improved/",
                       "../src_data/train_features/OMV62of65_npy_shot_40aco_improved/"]

            after_foot = "_shot_40aco.npy"

        elif output_feature_type == "mean_and_std":
            to_dirs = ["../src_data/train_features/OMV200_npy_shot_80aco/",
                       "../src_data/train_features/OMV62of65_npy_shot_80aco/"]

            after_foot = "_shot_80aco.npy"

    for category_itr in range(2):

        # videoXのnpyを読み込む
        for video_index, frame_feat_file in enumerate(os.listdir(from_feat_dirs[category_itr])):

            frame_features = np.load(from_feat_dirs[category_itr] + frame_feat_file)

            tmp_video_name = frame_feat_file.split(before_foot)
            video_name = tmp_video_name[0]  # video00001
            shot_img_dir = root_shot_img_dirs[category_itr] + video_name

            shot_features = np.array([])

            # videoXの各フレームを読み込む
            if len(os.listdir(shot_img_dir)) > threshold:  # ショット検出できていない動画があるので、その動画は扱わない

                prev_row_index = 0
                now_row_index = 0

                for img_index, img_file in enumerate(os.listdir(shot_img_dir)):

                    # ファイルの名前をもとに特徴量をまとめる
                    tmp_shot_posi = img_file.split('_')  # ->[video00001, 01707, 71.196124.png]
                    tmp_shot_posi = tmp_shot_posi[3].split('.png')  # ->[71.196124]
                    shot_posi = float(tmp_shot_posi[0])  # 71.196124

                    if img_index > 0:  # 最初のフレームに対して計算することを防ぐ
                        while now_row_index * frame_hop + frame_length <= shot_posi:
                            now_row_index += 1

                        this_shot_feat = frame_features[prev_row_index:now_row_index, :]
                        tmp_shot_features = calc_max_min_and_concatenate_features(this_shot_feat, output_feature_type)

                        prev_row_index = now_row_index

                        if shot_features.size == 0:
                            shot_features = tmp_shot_features
                        else:
                            shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

                # 最後のショットの情報をここで得る
                this_shot_feat = frame_features[now_row_index:, :]
                tmp_shot_features = calc_max_min_and_concatenate_features(this_shot_feat, output_feature_type)

                shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

                # 例外処理
                if not (all(math.isfinite(shot_features[i][j]))
                        for i in range(shot_features.shape[0])
                        for j in range(shot_features.shape[1])):
                    sys.stderr.write("nan comes!!")
                    print(shot_features)
                    sys.exit()

                # npy保存
                npy_name = to_dirs[category_itr] + video_name + after_foot

                np.save(npy_name, shot_features)

                print(shot_features.shape)
                print(video_index + 1, '曲目が終了')


if __name__ == "__main__":

    # for_training_convert_frame2shot_features(10)
    for_test_convert_frame2shot_features(10)