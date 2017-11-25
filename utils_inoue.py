import csv
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os
import re

from math import sqrt
from sklearn.metrics import mean_squared_error
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
def csv2npy(from_dir, to_dir, threshold):

    for index, file in enumerate(os.listdir(from_dir)):
        # csvの読み込み
        data = np.loadtxt(from_dir+file, delimiter=',')
        if np.ndim(data) == 2:
            if data.shape[0] > threshold:
                print(index, data.shape)
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
                npy_name = to_dir + filename+'.npy'
                np.save(npy_name, data)

    #
    # load_file = np.load('./data/aco_features/video2.npy')
    # print(load_file)


def for_test_convert_frame2shot_features():

    frame_feat_dir = '../src_data/recommendation_test_features/csv_frame_46aco/'
    root_shot_img_dir = '../src_data/shots_recommendation_test/'

    for test_vid_itr, test_vid_folder in enumerate(os.listdir(root_shot_img_dir)):

        shot_img_dir = root_shot_img_dir + test_vid_folder
        # csv_to_dir = '../src_data/recommendation_test_features/shot_features/'\
        #          + 'cut_by_' + test_vid_folder + '/'
        npy_to_dir = '../src_data/recommendation_test_features/npy_shot_46aco/cut_by_' + test_vid_folder + '/'

        # フレーム特徴量のcsvを読み込む
        for video_index, frame_feat_file in enumerate(os.listdir(frame_feat_dir)):
            tmp_music_name = frame_feat_file.split('.csv')
            music_name = tmp_music_name[0]  # test_music_00123

            csv_obj = csv.reader(open(frame_feat_dir + frame_feat_file, 'r'))
            frame_features = np.array([v for v in csv_obj])
            frame_features = frame_features.astype(np.float64)

            shot_features = np.array([])
            is_first_convert = True

            # videoXの各フレームを読み込む

            prev_col_index = 0
            now_col_index = 0

            for img_file in os.listdir(shot_img_dir):

                # ファイルの名前をもとに特徴量をまとめる
                tmp_shot_posi = img_file.split('_')  # ->[test,video00005,01906,63.596867.png]
                tmp_shot_posi = tmp_shot_posi[3].split('.png')  # -> [63.59687]
                shot_posi = float(tmp_shot_posi[0])  # 63.59687

                if shot_posi > 0.034:  # 最初のフレームは0.00333秒に現れるので、最初のフレームに対して計算することを防ぐ
                    while now_col_index*0.01 + 0.03 <= shot_posi:
                        now_col_index += 1

                    tmp_shot_features = np.mean(frame_features[prev_col_index:now_col_index, :], axis=0)
                    tmp_shot_features = tmp_shot_features.reshape(1, len(tmp_shot_features))
                    prev_col_index = now_col_index

                    if is_first_convert:
                        shot_features = tmp_shot_features
                        is_first_convert = False
                    else:
                        shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

            # 最後のショットの情報をここで得る
            tmp_shot_features = np.mean(frame_features[now_col_index:, :], axis=0)
            tmp_shot_features = tmp_shot_features.reshape(1, len(tmp_shot_features))
            shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

            npy_name = npy_to_dir + music_name + '_cut_by_' + test_vid_folder + '.npy'
            np.save(npy_name, shot_features)

            print(video_index+1, '曲目が終了')
            print(shot_features.shape)


def for_training_convert_frame2shot_features():

    frame_feat_dir = '../src_data/train_features/csv_46aco_frame_62of65/'
    root_shot_img_dir = '../src_data/shots_OMV62of65/'
    to_dir = '../src_data/train_features/npy_shot_46aco_62of65/'

    # videoXのcsvを読み込む
    for video_index, frame_feat_file in enumerate(os.listdir(frame_feat_dir)):
        tmp_video_name = frame_feat_file.split('_46aco_frame.csv')
        video_name = tmp_video_name[0]  # video00001

        shot_img_dir = root_shot_img_dir + video_name

        csv_obj = csv.reader(open(frame_feat_dir + frame_feat_file, 'r'))
        frame_features = np.array([v for v in csv_obj])
        frame_features = frame_features.astype(np.float64)

        frame_features = np.transpose(frame_features)

        shot_features = np.array([])
        is_first_convert = True

        # videoXの各フレームを読み込む
        if len(os.listdir(shot_img_dir)) > 5:  # ショット検出できていない動画があるので、その動画は扱わない

            prev_col_index = 0
            now_col_index = 0

            for img_file in os.listdir(shot_img_dir):

                # ファイルの名前をもとに特徴量をまとめる
                tmp_shot_posi = img_file.split('_')  # ->[video00001, 01707, 71.196124.png]
                tmp_shot_posi = tmp_shot_posi[2].split('.png')  # ->[71.196124]
                shot_posi = float(tmp_shot_posi[0])  # 71.196124

                if shot_posi > 0.045:  # 最初のフレームは0.00333秒に現れるので、最初のフレームに対して計算することを防ぐ
                    while now_col_index*0.01 + 0.03 <= shot_posi:
                        now_col_index += 1

                    tmp_shot_features = np.mean(frame_features[prev_col_index:now_col_index, :], axis=0)
                    tmp_shot_features = tmp_shot_features.reshape(1, len(tmp_shot_features))
                    prev_col_index = now_col_index

                    if is_first_convert:
                        shot_features = tmp_shot_features
                        is_first_convert = False
                    else:
                        shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

            # 最後のショットの情報をここで得る
            tmp_shot_features = np.mean(frame_features[now_col_index:, :], axis=0)
            tmp_shot_features = tmp_shot_features.reshape(1, len(tmp_shot_features))
            shot_features = np.concatenate([shot_features, tmp_shot_features], axis=0)

            # npy保存
            npy_name = to_dir + video_name + '_46_shot.npy'
            np.save(npy_name, shot_features)

            print(shot_features.shape)
            print(video_index+1, '曲目が終了')


def identify_data_order():

    x_dir = './data/shot_features/4608hsv_features/'
    y_dir = './data/shot_features/aco_features/'

    x_order = np.array([])
    y_order = np.array([])

    for x_index, x_file in enumerate(os.listdir(x_dir)):
        data = np.load(x_dir+x_file)

        if np.ndim(data) == 2:
            if data.shape[0] > 10:
                tmp_order = np.array([data.shape[0]])
                if x_index == 0:
                    x_order = tmp_order
                else:
                    x_order = np.concatenate([x_order, tmp_order])

    for y_index, y_file in enumerate(os.listdir(y_dir)):
        data = np.load(y_dir+y_file)
        tmp_order = np.array([data.shape[0]])
        if y_index == 0:
            y_order = tmp_order
        else:
            y_order = np.concatenate([y_order, tmp_order])

    print(len(x_order))
    print(len(y_order))

    if len(x_order) != len(y_order):
        print('the number of data of x and that of y is different!!')
        print('the number of data of x is ', len(x_order))
        print('the number of data of y is ', len(y_order))
    else:
        for i in range(len(x_order)):
            # print(x_order[i], y_order[i])
            if x_order[i] != y_order[i]:
                print('order', i, ' is not correct!')
                print(x_order[i], y_order[i])


# すでに取得しているmfcc,chroma,spectral contrastをつなげる
def make_46_aco_features():

    to_dir = '../src_data/recommendation_test_features/frame_46features/'

    chroma_path = '../src_data/recommendation_test_features/frame_chroma/'
    mfcc_path = '../src_data/recommendation_test_features/frame_20mfcc/'
    speccontrast_path = '../src_data/recommendation_test_features/frame_spect_contrast/'

    chroma_files = os.listdir(chroma_path)
    mfcc_files = os.listdir(mfcc_path)
    speccontrast_files = os.listdir(speccontrast_path)

    if len(chroma_files) != len(mfcc_files) or len(chroma_files) != len(speccontrast_files):
        print('These features size is different!')
        print(len(chroma_files))
        print(len(mfcc_files))
        print(len(speccontrast_files))
    else:
        for itr in range(len(chroma_files)):
            # つなげる
            chroma = np.loadtxt(chroma_path + chroma_files[itr], delimiter=',')
            mfcc = np.loadtxt(mfcc_path + mfcc_files[itr], delimiter=',')
            speccontrast = np.loadtxt(speccontrast_path + speccontrast_files[itr], delimiter=',')

            if speccontrast.shape[0] > mfcc.shape[0]:
                print('diff speccontrast - mfcc:', speccontrast.shape[0] - mfcc.shape[0])
                speccontrast = speccontrast[:mfcc.shape[0], :]
            elif speccontrast.shape[0] < mfcc.shape[0]:
                print('diff mfcc - speccontrast:', mfcc.shape[0] - speccontrast.shape[0])
                chroma = chroma[:speccontrast.shape[0], :]
                mfcc = mfcc[:speccontrast.shape[0], :]

            features = np.concatenate([chroma, mfcc, speccontrast], axis=1)
            print(features.shape)
            if features.shape[1] != 46:
                print("This feature's size is not 46!!")
                print(chroma.shape)
                print(mfcc.shape)
                print(speccontrast.shape)
            else:
                name_index = ''
                if itr+1 < 10:
                    name_index = '0000' + str(itr+1)
                elif itr + 1 < 100:
                    name_index = '000' + str(itr + 1)
                elif itr + 1 < 1000:
                    name_index = '00' + str(itr + 1)

                csv_name = to_dir + 'test_music_'+name_index + '_46frame.csv'
                np.savetxt(csv_name, features, delimiter=',')


# ディレクトリを作成
def make_directory(to_dir):

    for i in range(18):

        file_index = ""
        if i + 1 < 10:
            file_index = "0000"+str(i + 1)
        elif i + 1 < 100:
            file_index = "000" + str(i + 1)
        elif i + 1 < 1000:
            file_index = "00" + str(i + 1)
        elif i + 1 < 10000:
            file_index = "0" + str(i + 1)
        else:
            file_index = str(i + 1)

        path = to_dir + "cut_by_test_video" + file_index
        os.mkdir(path)


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

    from_dir = '../src_data/OMV_62of65/old_OMV_62of65/'
    to_dir = '../src_data/OMV_62of65/OMV_62of65/'

    name_index = 0

    correspond_name = np.array([])

    for index, file in enumerate(os.listdir(from_dir)):
        # mp3 = re.compile('mp3')
        # m4a = re.compile('m4a')
        mp4 = re.compile('mp4')

        # if mp3.search(file) or m4a.search(file):
        if mp4.search(file):

            if name_index+1 < 10:
                filename = 'video0000' + str(name_index+1)
            elif name_index+1 < 100:
                filename = 'video000' + str(name_index+1)
            elif name_index+1 < 1000:
                filename = 'video00' + str(name_index+1)
            elif name_index+1 < 10000:
                filename = 'video0' + str(name_index+1)
            else:
                filename = 'video' + str(name_index+1)

            # if mp3.search(file):
            #     os.rename(from_dir + file, './data/recommendation_test/music/' + filename + '.mp3')
            # elif m4a.search(file):
            #     os.rename(from_dir + file, './data/recommendation_test/music/' + filename + '.m4a')

            os.rename(from_dir + file, to_dir + filename + '.mp4')

            print(file, filename)

            # 以前の名前と変更後の名前の対応表をcsvで作る
            tmp_corr_name = np.array([[file, filename]])
            if name_index == 0:
                correspond_name = tmp_corr_name
            else:
                correspond_name = np.concatenate([correspond_name, tmp_corr_name], axis=0)

            name_index += 1

        else:
            pass

    # csv保存
    csv_name = to_dir + 'correspond_original_name.csv'
    with open(csv_name, 'w') as f:
        writer = csv.writer(f, lineterminator='/n')
        writer.writerows(correspond_name)


# 名前の対応表をもとにファイルの名前を変更
def rename2():

    from_dir = '../src_data/OMV_62of65/old_OMV_62of65/'
    to_dir = '../src_data/OMV_62of65/OMV_62of65/'

    correspond_table_path = '../src_data/OMV_62of65/OMV_62of65/correspond_original_name.csv'
    csv_obj = csv.reader(open(correspond_table_path, 'r'))
    correspond_table = np.array([v for v in csv_obj])

    for src_file in os.listdir(from_dir):
        tmp_src_name = src_file.split('.csv')
        src_name = tmp_src_name[0]

        for ref_index in range(correspond_table.shape[0]):

            tmp_ref_name = correspond_table[ref_index, 0].split('.mp4')
            ref_name = tmp_ref_name[0]

            # 同じ名前だったらreanameする、同じでなければそのまま
            if src_name == ref_name:
                print(src_name, '  ', correspond_table[ref_index, 1])
                os.rename(from_dir + src_file, to_dir + correspond_table[ref_index, 1] + '_bar.csv')


def save_mean_std_max_min(feature_name, from_dirs, to_dir):

    target_arr = np.array([])

    for entire_index, directory in enumerate(from_dirs):
        for local_index, file in enumerate(os.listdir(directory)):
            tmp_target_array = np.load(directory + file)
            if np.ndim(tmp_target_array) == 2 and tmp_target_array.shape[0] > 10:
                if entire_index == 0 and local_index == 0:
                    target_arr = tmp_target_array
                else:
                    target_arr = np.concatenate([target_arr, tmp_target_array], axis=0)

            print(entire_index+1, ' ', local_index+1)

    mean_1d = np.mean(target_arr, axis=0)
    mean_2d = np.matlib.repmat(mean_1d, target_arr.shape[0], 1)
    std_1d = np.std(target_arr, axis=0)
    tmp_std_min = np.sort(std_1d)
    std_min_nonzero = None
    for min_itr in range(len(tmp_std_min)):
        if tmp_std_min[min_itr] > 0:
            std_min_nonzero = tmp_std_min[min_itr]
            break

    print(std_min_nonzero)

    for itr in range(len(std_1d)):
        if std_1d[itr] == 0:
            std_1d[itr] = std_min_nonzero

    std_2d = np.matlib.repmat(std_1d, target_arr.shape[0], 1)

    standardized_arr = (target_arr - mean_2d) / std_2d

    max_1d = np.max(standardized_arr, axis=0)
    min_1d = np.min(standardized_arr, axis=0)

    for itr in range(len(max_1d)):
        if max_1d[itr] == min_1d[itr]:
            print(itr, ' are the same!')
            max_1d[itr] += 0.1
            min_1d[itr] -= 0.1

    mean_std_max_min = np.concatenate([mean_1d.reshape(1, target_arr.shape[1]),
                                       std_1d.reshape(1, target_arr.shape[1]),
                                       max_1d.reshape(1, target_arr.shape[1]),
                                       min_1d.reshape(1, target_arr.shape[1])], axis=0)

    np.save(to_dir + 'shot_' + feature_name + '_mean_std_max_min.npy', mean_std_max_min)


if __name__ == '__main__':

    # concatenate_file(from_dir='./data/fc7_features/before/video42/', to_dir='./data/fc7_features/after/dummy/', axis=1)
    #
    # csv2npy(from_dir='../src_data/train_features/csv_shot_768bgr/',
    #         to_dir='../src_data/train_features/npy_shot_768bgr/', threshold=10)
    # for_test_convert_frame2shot_features()
    # for_train_convert_frame2shot_features()
    # identify_data_order()
    # make_46_aco_features()
    # make_directory("../src_data/recommendation_test_features/npy_shot_46aco/")
    # rename()
    # rename2()
    save_mean_std_max_min(feature_name='46aco',
                          from_dirs=('../src_data/train_features/npy_shot_46aco/',
                                     '../src_data/recommendation_test_features/npy_shot_46aco/cut_by_test_video00001/',),
                          to_dir='./output/min_max_mean_std/')

    # root_dir = '../src_data/recommendation_test_features/frame_77features/'
    # chroma_dir = '../src_data/recommendation_test_features/frame_chroma/'
    # mfcc_dir = '../src_data/recommendation_test_features/frame_20mfcc/'
    # for index, file in enumerate(os.listdir(root_dir)):
    #     frame_features = np.loadtxt(root_dir+file, delimiter=',')
    #     chroma = frame_features[:, 4:16]
    #     mfcc = frame_features[:, 16:36]
    #
    #     print(chroma.shape)
    #     print(mfcc.shape)
    #
    #     name_index = ''
    #     if index+1 < 10:
    #         name_index = '0000' + str(index+1)
    #     elif index+1 < 100:
    #         name_index = '000' + str(index+1)
    #     elif index + 1 < 1000:
    #         name_index = '00' + str(index+1)
    #
    #     # csv保存
    #     chroma_name = chroma_dir + 'test_music_' + name_index + '_chroma_frame.csv'
    #     np.savetxt(chroma_name, chroma, delimiter=',')
    #
    #     mfcc_name = mfcc_dir + 'test_music_' + name_index + '_20mfcc_frame.csv'
    #     np.savetxt(mfcc_name, mfcc, delimiter=',')