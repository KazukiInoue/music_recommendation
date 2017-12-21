import math
import numpy as np
import os
import sys


def make_histogram(hist_src, max_arr, min_arr):

    hist_count = np.array([])

    for feat_itr in range(hist_src.shape[1]):

        src = hist_src[:, feat_itr]
        max_val = max_arr[feat_itr]
        min_val = min_arr[feat_itr]
        tmp_hist_count, hist_mean = np.histogram(src, range=[min_val, max_val], bins=5)  # bins=23では良い結果が得られなかったので、ビンの数を減らした

        if hist_count.size == 0:
            hist_count = tmp_hist_count
        else:
            hist_count = np.concatenate([hist_count, tmp_hist_count], axis=0)

    return hist_count


def calc_and_concatenate_features(src_array, max_arr, min_arr):

    mean_features = np.mean(src_array, axis=0)
    std_features = np.std(src_array, axis=0)
    hist_features = make_histogram(src_array, max_arr, min_arr)

    out_arr = np.concatenate([mean_features, std_features, hist_features], axis=0)
    out_arr = out_arr.reshape(1, len(out_arr))

    return out_arr


def for_training_add_histogram_and_convert_frame2shot_features(threshold):

    feature_type = "40aco"

    root_shot_img_dirs = ["../src_data/shots_OMV200_improved/",  # timposiとの兼ね合いがあるので,必ず0番目の要素にOMV200の情報を入れること
                          "../src_data/shots_OMV62of65_improved/"]

    from_feat_dirs = ["", ""]
    to_dirs = ["", ""]
    before_foot = ""
    after_foot = ""
    frame_length = 0
    frame_hop = 0

    if feature_type == "46aco":

        from_feat_dirs = ["../src_data/train_features/OMV200_npy_frame_46aco/",
                          "../src_data/train_features/OMV62of65_npy_frame_46aco/"]

        to_dirs = ["../src_data/train_features/OMV200_npy_shot_322aco/",
                   "../src_data/train_features/OMV62of65_npy_shot_322aco/"]

        before_foot = "_frame_46aco.npy"
        after_foot = "_shot_322aco.npy"
        frame_length = 0.03
        frame_hop = 0.01

    elif feature_type == "40aco":

        from_feat_dirs = ["../src_data/train_features/OMV200_npy_frame_40aco/",
                          "../src_data/train_features/OMV62of65_npy_frame_40aco/"]

        to_dirs = ["../src_data/train_features/OMV200_npy_shot_280aco/",
                   "../src_data/train_features/OMV62of65_npy_shot_280aco/"]

        before_foot = "_frame_40aco.npy"
        after_foot = "_shot_280aco.npy"
        frame_length = 2048 / 44100
        frame_hop = 512 / 44100

    print("最大値と最小値を計算中...")

    all_features = np.array([])

    for from_feat_dir in from_feat_dirs:
        for file in os.listdir(from_feat_dir):
            tmp_features = np.load(from_feat_dir + file)
            if all_features.size == 0:
                all_features = tmp_features
            else:
                all_features = np.concatenate([all_features, tmp_features], axis=0)

    max_arr = np.max(all_features, axis=0)
    min_arr = np.min(all_features, axis=0)

    for category_itr in range(2):

        # videoXのnpyを読み込む
        for video_index, frame_feat_file in enumerate(os.listdir(from_feat_dirs[category_itr])):

            frame_features = np.load(from_feat_dirs[category_itr] + frame_feat_file)

            tmp_video_name = frame_feat_file.split(before_foot)
            video_name = tmp_video_name[0]  # video00001
            shot_img_dir = root_shot_img_dirs[category_itr] + video_name

            shot_features = np.array([])

            # videoXの各フレームを読み込む
            if len(os.listdir(shot_img_dir)) > threshold:  # ショット数が少ない動画があるので、その動画は扱わない

                prev_row_index = 0
                now_row_index = 0

                for img_index, img_file in enumerate(os.listdir(shot_img_dir)):

                    # ファイルの名前をもとに特徴量をまとめる
                    tmp_shot_posi = img_file.split('_')  # ->[video00001, 01707, 71.196124.png]
                    tmp_shot_posi = tmp_shot_posi[3].split('.png')  # ->[71.196124]
                    shot_posi = float(tmp_shot_posi[0])  # 71.196124

                    if img_index > 0:  # 最初のフレームに対して計算することを防ぐ
                        while now_row_index*frame_hop + frame_length <= shot_posi:
                            now_row_index += 1

                        this_shot_feat = frame_features[prev_row_index:now_row_index, :]
                        tmp_shot_features = calc_and_concatenate_features(this_shot_feat, max_arr, min_arr)

                        prev_row_index = now_row_index

                        if shot_features.size == 0:
                            shot_features = tmp_shot_features
                        else:
                            shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

                # 最後のショットの情報をここで得る
                this_shot_feat = frame_features[now_row_index:, :]
                tmp_shot_features = calc_and_concatenate_features(this_shot_feat, max_arr, min_arr)

                shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

                # 例外処理
                if not(all(math.isfinite(shot_features[i][j]))
                       for i in range(shot_features.shape[0])
                       for j in range(shot_features.shape[1])):

                    sys.stderr.write("nan comes!!")
                    print(shot_features)
                    sys.exit()

                # npy保存
                npy_name = to_dirs[category_itr] + video_name + after_foot

                np.save(npy_name, shot_features)

                print(shot_features.shape)
                print(video_index+1, '曲目が終了')


if __name__ == "__main__":

    for_training_add_histogram_and_convert_frame2shot_features(threshold=10)