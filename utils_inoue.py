import csv
import math
import numpy as np
import os
import re
import sys


# csvファイルからnpyファイルへの変換
def csv2npy(from_dir, to_dir, threshold):

    for index, file in enumerate(os.listdir(from_dir)):
        # csvの読み込み
        data = np.loadtxt(from_dir+file, delimiter=',')
        if np.ndim(data) == 2 and data.shape[0] > threshold:
            print(index, data.shape)

            tmp_filename = file.split('.csv')
            filename = tmp_filename[0]
            npy_name = to_dir + filename+'.npy'
            np.save(npy_name, data)

    #
    # load_file = np.load('./data/aco_features/video2.npy')
    # print(load_file)


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


# ディレクトリを作成
def make_directory(to_dir):

    for i in range(133):

        file_index = ""
        index_val = 1

        if i + index_val < 10:
            file_index = "0000"+str(i + index_val)
        elif i + index_val < 100:
            file_index = "000" + str(i + index_val)
        elif i + index_val < 1000:
            file_index = "00" + str(i + index_val)
        elif i + index_val < 10000:
            file_index = "0" + str(i + index_val)
        else:
            file_index = str(i + index_val)

        path = to_dir + "IMV133_video" + file_index
        os.mkdir(path)


# ファイル名の変更
def rename():

    from_dir = "../src_data/train_features/csv_frame_20cepstrum/"
    to_dir = "../src_data/train_features/IMV133_csv_frame_20cepstrum/"

    index_num = 1
    head_name = "IMV133_"
    extension = "csv"

    correspond_name = np.array([])

    for index, file in enumerate(os.listdir(from_dir)):
        # mp3 = re.compile('mp3')
        # m4a = re.compile('m4a')

        # if mp3.search(file) or m4a.search(file):
        if re.compile(extension).search(file):

            # if index+index_num < 10:
            #     filename = head_name + 'video0000' + str(index+index_num)
            # elif index+index_num < 100:
            #     filename = head_name + 'video000' + str(index+index_num)
            # elif index+index_num < 1000:
            #     filename = head_name + 'video00' + str(index+index_num)
            # elif index+index_num < 10000:
            #     filename = head_name + 'video0' + str(index+index_num)
            # else:
            #     filename = head_name + 'video' + str(index+index_num)

            # if mp3.search(file):
            #     os.rename(from_dir + file, './data/recommendation_test/music/' + filename + '.mp3')
            # elif m4a.search(file):
            #     os.rename(from_dir + file, './data/recommendation_test/music/' + filename + '.m4a')


            # os.rename(from_dir + file, to_dir + filename + '.' + extension)

            filename = head_name + file
            os.rename(from_dir + file, to_dir + filename)

            print(file, filename)

            # 以前の名前と変更後の名前の対応表をcsvで作る
            tmp_corr_name = np.array([[file, filename]])
            if correspond_name.size == 0:
                correspond_name = tmp_corr_name
            else:
                correspond_name = np.concatenate([correspond_name, tmp_corr_name], axis=0)

        else:
            pass

    # csv保存
    # csv_name = to_dir + 'correspond_table_of_name.csv'
    # with open(csv_name, 'w') as f:
    #     writer = csv.writer(f, lineterminator='\n')
    #     writer.writerows(correspond_name)


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


if __name__ == '__main__':

    # csv2npy(from_dir='../src_data/train_features/csv_frame_46aco_12chroma_20mfcc_14spectcontrast/',
    #         to_dir='../src_data/train_features/npy_frame_46aco/', threshold=10)


    # for_test_convert_frame2shot_features()
    # for_training_convert_frame2shot_features(threshold=10)
    # identify_data_order()
    # concatenate_multiple_features()
    make_directory("C/MUSIC_RECOMMENDATION/src_data/shots_IMV133_threshold=130_themes=6/")
    # rename()
    # rename2()
    #
    # from_root_dir = "../src_data/shots_train_threshold_themes/"
    # to_root_dir = "../src_data/shots_IMV133_threshold_themes/"
    # for from_dir, to_dir in zip(os.listdir(from_root_dir), os.listdir(to_root_dir)):
    #     for file in os.listdir(from_root_dir+from_dir):
    #
    #         to_name = "IMV133_"+file
    #         os.rename(from_root_dir+from_dir+"/"+file, to_root_dir+to_dir+"/"+to_name)
