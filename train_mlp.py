import csv
import math
import numpy as np
import numpy.matlib
import os
import sys

from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib


# ファイルの読み込みと曲の区切れ目を変えす
def load_file(from_dir):
    out_data = np.array([])

    for iteration, file in enumerate(os.listdir(from_dir)):
        data = np.load(from_dir + file)
        if np.ndim(data) == 2 and data.shape[0] > 10:
            if out_data.size == 0:
                out_data = data
            else:
                out_data = np.concatenate([out_data, data], axis=0)

    return out_data


def record_music_begin_end(max_data_num, from_dir):
    begin_end = np.array([])
    end = 0

    for iteration, file in enumerate(os.listdir(from_dir)):
        data = np.load(from_dir + "/" + file)
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

    mean_std_max_min = np.load("./output/mean_std_max_min/shot_" + feature_name + "_mean_std_max_min.npy")

    if len(in_array.shape) == 2:

        mean_2d = np.matlib.repmat(mean_std_max_min[0, :], in_array.shape[0], 1)
        std_2d = np.matlib.repmat(mean_std_max_min[1, :], in_array.shape[0], 1)
        max_2d = np.matlib.repmat(mean_std_max_min[2, :], in_array.shape[0], 1)
        min_2d = np.matlib.repmat(mean_std_max_min[3, :], in_array.shape[0], 1)

        # 標準化
        standardized_arr = (in_array - mean_2d)/std_2d

        # 正規化
        standed_normed_arr = (standardized_arr - min_2d) / (max_2d - min_2d)

        assert(all(np.max(standed_normed_arr, axis=0)[v] <= 1 for v in range(standed_normed_arr.shape[1])))
        assert (all(np.min(standed_normed_arr, axis=0)[v] >= 0 for v in range(standed_normed_arr.shape[1])))

        if not (all(math.isfinite(standed_normed_arr[i][j]))
                for i in range(standed_normed_arr.shape[0])
                for j in range(standed_normed_arr.shape[1])):
            sys.stderr.write("In ", feature_name, "nan comes!!")
            sys.exit()

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

    # 使用する特徴量の指定
    video_feat_types = ["768bgr",
                        "768hsv",
                        # "768lab",
                        # "60color_themes_bgr",
                        # "828bgr_themes",
                        # "828hsv_themes", "828lab_themes"
                        "512bgr", "512hsv",
                        # "512lab"
                        ]
    aco_feat_types = ["40aco",  # "46aco",
                      "280aco",
                      # "322aco", "92aco",
                      "80aco"
                      ]

    est_type = "video2aco"

    params = [{"solver": "lbfgs", "learning_rate": "constant"},
              # {"solver": "lbfgs", "learning_rate": "invscaling"},
              # {"solver": "lbfgs", "learning_rate": "adaptive"},
              # {"solver": "sgd",   "learning_rate": "constant"},
              # {"solver": "sgd",   "learning_rate": "invscaling"},
              # {"solver": "sgd",   "learning_rate": "adaptive"},
              # {"solver": "adam",  "learning_rate": "constant"},
              # {"solver": "adam",  "learning_rate": "invscaling"},
              # {"solver": "adam",  "learning_rate": "adaptive"},
              ]

    labels = ["lbfgs with constant learning-rate",
              # "lbfgs with inv-scaling learning-rate",
              # "lbfgs with adaptive learning-rate",
              # "sgd with constant learning-rate",
              # "sgd with inv-scaling learning-rate",
              # "sgd with adaptive learning-rate",
              # "adam with constant learning-rate",
              # "adam with inv-scaling learning-rate",
              # "adam with adaptive learning-rate",
              ]

    for video_feat_type in video_feat_types:
        for aco_feat_type in aco_feat_types:

            print("%s and %s" % (video_feat_type, aco_feat_type))

            dir_x = "../src_data/train_features/OMV200_npy_shot_" + video_feat_type + "/"
            dir_y = "../src_data/train_features/OMV200_npy_shot_" + aco_feat_type + "/"

            video_features = load_file(from_dir=dir_x)
            aco_features = load_file(from_dir=dir_y)

            # 正規化
            video_feat_norm = standardize_normalize_array(video_features, video_feat_type)
            aco_feat_norm = standardize_normalize_array(aco_features, aco_feat_type)

            X = np.array([])
            Y = np.array([])
            if est_type == "video2aco":
                X = video_feat_norm
                Y = aco_feat_norm
            elif est_type == "aco2video":
                X = aco_feat_norm
                Y = video_feat_norm

            print(X.shape)
            print(Y.shape)

            # for each dataset, plot learning for each learning strategy
            mlps = []
            max_iter = 10000
            score_loss_table = np.array([[""], ["Training set score"], ["Training set less"]])

            for label, param in zip(labels, params):

                print("training: %s" % label)
                mlp = MLPRegressor(activation="relu", hidden_layer_sizes=230, max_iter=max_iter, **param, random_state=1)
                mlp.fit(X, Y)
                mlps.append(mlp)
                print("Training set score: %f" % mlp.score(X, Y))
                print("Training set loss: %f" % mlp.loss_)
                print("")

                tmp_score_loss_table = np.array([[label], [str(mlp.score(X, Y))], [str(mlp.loss_)]])
                score_loss_table = np.concatenate([score_loss_table, tmp_score_loss_table], axis=1)

                if est_type == "video2aco":
                    joblib.dump(mlp, "./output/output_model/mlp_maxiter=" + str(max_iter) + "_" + param["solver"] + "_" + param["learning_rate"]
                                + "_230_OMV200_shot_" + video_feat_type + "_2_" + aco_feat_type + ".pkl")
                elif est_type == "aco2video":
                    joblib.dump(mlp, "./output/output_model/mlp_maxiter=" + str(max_iter) + "_" + param["solver"] + "_" + param["learning_rate"]
                                + "_230_OMV200_shot_" + aco_feat_type + "_2_" + video_feat_type + ".pkl")

            table_name = ""
            if est_type == "video2aco":
                table_name = "./output/score_and_loss/OMV200_230_score_and_loss_maxiter=" \
                             + str(max_iter) + "_" + video_feat_type + "_2_" + aco_feat_type + ".csv"
            elif est_type == "aco2video":
                table_name = "./output/score_and_loss/OMV200_230_score_and_loss_maxiter=" \
                             + str(max_iter) + "_" + video_feat_type + "_2_" + aco_feat_type + ".csv"

            with open(table_name, "w") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(score_loss_table)


if __name__ == "__main__":
    main()