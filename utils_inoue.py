import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os
import re
import time

from math import sqrt
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.svm import SVR


# npyファイルの統合
def concatenate_file(from_dir, to_dir, axis):

    for folder in os.listdir(from_dir):
        is_first = True
        out_array = np.array([])
        video_index = []
        for file in os.listdir(from_dir+folder):
            data = np.load(from_dir+folder+file)
            if is_first:
                out_array = data
                video_index = file.split('_')
                is_first = False
            else:
                out_array = np.concatenate([out_array, data], axis=axis)

        # バイナリファイル保存
        binary_name = to_dir+'/'+video_index[0]
        np.save(binary_name, out_array)
        print('video_index[0] = ', video_index[0])


# csvファイルからnpyファイルへの変換
def csv2npy(from_dir, to_dir):

    for index, file in enumerate(os.listdir(from_dir)):
        # csvの読み込み
        data = np.loadtxt(from_dir+file, delimiter=',')
        print('data.shape = ', data.shape)
        # if index + 1 < 10:
        #     filename = 'video0000'+str(index+1)+'.npy'
        # elif index + 1 < 100:
        #     filename = 'video000' + str(index+1) + '.npy'
        # elif index + 1 < 1000:
        #     filename = 'video00' + str(index+1) + '.npy'
        # elif index + 1 < 10000:
        #     filename = 'video0' + str(index+1) + '.npy'
        # else:
        #     filename = str(index) + '.npy'
        tmp_filename = file.split('.csv')
        filename = tmp_filename[0]
        npy_name = to_dir + filename
        np.save(npy_name, data)

    #
    # load_file = np.load('./data/aco_features/video2.npy')
    # print(load_file)


def identify_data_order():

    x_dir = './data/hsv_features/'
    y_dir = './data/test_aco_features/'

    x_order = np.array([])
    y_order = np.array([])

    for x_index, x_file in enumerate(os.listdir(x_dir)):
        if x_index == 130:
            break
        else:
            data = np.load(x_dir+x_file)
            tmp_order = np.array([data.shape[0]])
            if x_index == 0:
                x_order = tmp_order
            else:
                x_order = np.concatenate([x_order, tmp_order])

    for y_index, y_file in enumerate(os.listdir(y_dir)):
        if y_index == 130:
            break
        else:
            data = np.load(y_dir+y_file)
            tmp_order = np.array([data.shape[1]])
            if y_index == 0:
                y_order = tmp_order
            else:
                y_order = np.concatenate([y_order, tmp_order])

    print(len(x_order))
    print(len(y_order))

    for i in range(len(x_order)):
        # print(x_order[i], y_order[i])
        if x_order[i] != y_order[i]:
            print('order', i, ' is not correct!')
            print(x_order[i], y_order[i])

    print(np.sum(x_order))
    print(np.sum(y_order))


# npyファイルからグラフを作成
def npy2graph():

    data = np.load('C:/python_projects/svr/output_cv.npy/cv_1th~70th_aco_feature[0].npy')

    x = np.array(data[:, 0])
    y = np.array(data[:, 2])

    # 図を作成
    plt.figure(figsize=[10, 5])
    plt.hold('on')
    plt.plot(x, y, c='b', label='RBF model')
    plt.xlabel('iteration')
    plt.ylabel('Root Mean Square Error')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


# ファイル名の変更
def rename():

    from_dir = './data/recommendation_test/old_music/'

    name_index = 0

    for index, file in enumerate(os.listdir(from_dir)):
        mp3 = re.compile('mp3')
        m4a = re.compile('m4a')

        if mp3.search(file) or m4a.search(file):

            if name_index < 10:
                filename = 'test_music_0000' + str(name_index)
            elif name_index < 100:
                filename = 'test_music_000' + str(name_index)
            elif name_index < 1000:
                filename = 'test_music_00' + str(name_index)
            elif name_index < 10000:
                filename = 'test_music_0' + str(name_index)
            else:
                filename = 'test_music_' + str(name_index)

            name_index += 1

            if mp3.search(file):
                os.rename(from_dir + file, './data/recommendation_test/music/' + filename + '.mp3')
            elif m4a.search(file):
                os.rename(from_dir + file, './data/recommendation_test/music/' + filename + '.m4a')

            print(file, filename)

        else:
            pass


def save_mean_std_max_min(feature_name, from_dirs, to_dir):

    target_arr = np.array([])

    for entire_index, directory in enumerate(from_dirs):
        for local_index, file in enumerate(os.listdir(directory)):

            tmp_target_array = np.load(directory + file)

            if entire_index == 0 and local_index == 0:
                target_arr = tmp_target_array
            else:
                target_arr = np.concatenate([target_arr, tmp_target_array], axis=0)

            print(entire_index+1, ' ', local_index+1)

    mean_1d = np.mean(target_arr, axis=0)
    mean_2d = np.matlib.repmat(mean_1d, target_arr.shape[0], 1)
    std_1d = np.std(target_arr, axis=0)
    std_2d = np.matlib.repmat(std_1d, target_arr.shape[0], 1)

    standardized_arr = (target_arr - mean_2d) / std_2d

    max_1d = np.max(standardized_arr, axis=0)
    min_1d = np.min(standardized_arr, axis=0)

    mean_std_max_min = np.concatenate([mean_1d.reshape(1, target_arr.shape[1]),
                                       std_1d.reshape(1, target_arr.shape[1]),
                                       max_1d.reshape(1, target_arr.shape[1]),
                                       min_1d.reshape(1, target_arr.shape[1])], axis=0)

    np.save(to_dir + feature_name + '_mean_std_max_min.npy', mean_std_max_min)


def test_joblib_dump():

    # インプットを適当に生成
    X1 = np.sort(1 * np.random.rand(40, 1).reshape(40), axis=0)
    X2 = np.sort(3 * np.random.rand(40, 1).reshape(40), axis=0)
    X3 = np.sort(5 * np.random.rand(40, 1).reshape(40), axis=0)
    X4 = np.sort(7 * np.random.rand(40, 1).reshape(40), axis=0)

    # インプットの配列を一つに統合
    X = np.c_[X1, X2, X3, X4]
    #
    # X1 = np.sort(1 * np.random.rand(40, 1), axis=0)
    # X2 = np.sort(3 * np.random.rand(40, 1), axis=0)
    # X3 = np.sort(5 * np.random.rand(40, 1), axis=0)
    # X4 = np.sort(7 * np.random.rand(40, 1), axis=0)
    #
    # X = np.concatenate([X1, X2, X3, X4], axis=1)

    # アウトプットを算出
    y = np.sin(X1).ravel() + np.cos(X2).ravel() + np.sin(X3).ravel() - np.cos(X4).ravel()

    y_o = y.copy()

    # ノイズを加える
    y[::5] += 3 * (0.5 - np.random.rand(8))

    # フィッティング
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)

    svr_rbf.fit(X, y)
    svr_lin.fit(X, y)

    y_rbf = svr_rbf.predict(X)
    y_lin = svr_lin.predict(X)

    # test1
    # テストデータも準備
    test_X1 = np.sort(2 * np.random.rand(10, 1).reshape(10), axis=0)
    test_X2 = np.sort(4 * np.random.rand(10, 1).reshape(10), axis=0)
    test_X3 = np.sort(6 * np.random.rand(10, 1).reshape(10), axis=0)
    test_X4 = np.sort(8 * np.random.rand(10, 1).reshape(10), axis=0)

    test_X = np.c_[test_X1, test_X2, test_X3, test_X4]
    test_y = np.sin(test_X1).ravel() + np.cos(test_X2).ravel() + np.sin(test_X3).ravel() - np.cos(test_X4).ravel()

    # テストデータを突っ込んで推定してみる
    test_rbf = svr_rbf.predict(test_X)
    test_lin = svr_lin.predict(test_X)

    # 相関係数計算
    rbf_corr = np.corrcoef(test_y, test_rbf)[0, 1]
    lin_corr = np.corrcoef(test_y, test_lin)[0, 1]

    # RMSEを計算
    rbf_rmse = sqrt(mean_squared_error(test_y, test_rbf))
    lin_rmse = sqrt(mean_squared_error(test_y, test_lin))

    print('test_rbf =', test_rbf)

    print("RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))
    print("Linear: RMSE %f \t\t Corr %f" % (lin_rmse, lin_corr))
    print(test_rbf[0])

    # test2 # テストデータも準備
    test_X1 = np.sort(-2 * np.random.rand(20, 1).reshape(20), axis=0)
    test_X2 = np.sort(-1 * np.random.rand(20, 1).reshape(20), axis=0)
    test_X3 = np.sort(1 * np.random.rand(20, 1).reshape(20), axis=0)
    test_X4 = np.sort(2 * np.random.rand(20, 1).reshape(20), axis=0)

    test_X = np.c_[test_X1, test_X2, test_X3, test_X4]
    test_y = np.sin(test_X1).ravel() + np.cos(test_X2).ravel() + np.sin(test_X3).ravel() - np.cos(test_X4).ravel()

    # テストデータを突っ込んで推定してみる
    test_rbf = svr_rbf.predict(test_X)
    test_lin = svr_lin.predict(test_X)

    # 相関係数計算
    rbf_corr = np.corrcoef(test_y, test_rbf)[0, 1]
    lin_corr = np.corrcoef(test_y, test_lin)[0, 1]

    # RMSEを計算
    rbf_rmse = sqrt(mean_squared_error(test_y, test_rbf))
    lin_rmse = sqrt(mean_squared_error(test_y, test_lin))

    print()
    print("RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))
    print("Linear: RMSE %f \t\t Corr %f" % (lin_rmse, lin_corr))
    print(test_rbf[0])


if __name__ == '__main__':

    # save_mean_std_max_min()
    # concatenate_file(from_dir='./data/fc7_features/before/video42/', to_dir='./data/fc7_features/after/dummy/', axis=1)
    #
    # csv2npy(from_dir='C:/C++Projects/extract_video_features/output_video_features/4608hsv_features/train/',
    #         to_dir='C:/python_projects/music_recommendation/data/4608hsv_features/')

    save_mean_std_max_min(feature_name='4608hsv',
                          from_dirs=('./data/4608hsv_features/', './data/recommendation_test/4608hsv_features/'),
                          to_dir='./output/')