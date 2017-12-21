import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os
import sys

from sklearn.externals import joblib


# np.array(['a','b','c'])->np.array([['abc']])
def characters_into_one_element(characters, delimiter):
    tmp = delimiter.join(characters)
    tmp = np.array([tmp])
    characters_in_one_element = tmp.reshape(1, 1)

    return characters_in_one_element


# 配列の標準化と正規化を行う
def standardize_normalize_array(in_array, feature_name):

    mean_std_max_min = np.load('./output/mean_std_max_min/shot_' + feature_name + '_mean_std_max_min.npy')

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


def show_estimated_color_features():

    # 使用する音響特徴量の入力
    input_aco_type = "40aco"
    input_aco_path = ""
    input_aco = np.load(input_aco_path)
    aco_norm = standardize_normalize_array(input_aco, input_aco_type)

    # NNモデルに投入、色特徴量の出力
    est_feature_type = "828bgr_themes"

    model_dir = './output/output_model/mlp_lbfgs_adaptive_shot_80hsv_46aco_230_120_30.pkl'
    mlp = joblib.load(model_dir)
    est_color_features = mlp.predict(aco_norm)
    # 0~255までがB、256~511までがG、512~767までがRのヒストグラム、768827がカラーパレット
    if est_feature_type == "768bgr":
        blue_hist = est_color_features[:, 0:256]
        green_hist = est_color_features[:, 256:512]
        red_hist = est_color_features[:, 512:768]
    elif est_feature_type == "768hsv":
        hue_hist = est_color_features[:, 0:256]
        sat_hist = est_color_features[:, 256:512]
        val_hist = est_color_features[:, 512:768]
    if est_feature_type == "828bgr_themes":
        blue_hist = est_color_features[:, 0:256]
        green_hist = est_color_features[:, 256:512]
        red_hist = est_color_features[:, 512:768]
        color_themes = est_color_features[:, 768:828]
    elif est_feature_type == "828hsv_themes":
        hue_hist = est_color_features[:, 0:256]
        sat_hist = est_color_features[:, 256:512]
        val_hist = est_color_features[:, 512:768]
        color_themes = est_color_features[:, 768:828]

    bgr_vertical = np.arange(256)
    themes_vertical = np.arange(60)
    plt.bar(bgr_vertical, blue_hist)



if __name__ == '__main__':
    show_estimated_color_features()