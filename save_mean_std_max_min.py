import numpy as np
import numpy.matlib
import os


def save_mean_std_max_min(feature_name, from_dirs, to_dir):

    target_arr = np.array([])

    if type(from_dirs) == tuple:
        for entire_index, directory in enumerate(from_dirs):
            for local_index, file in enumerate(os.listdir(directory)):

                tmp_target_array = np.load(directory + file)

                if np.ndim(tmp_target_array) == 2 and tmp_target_array.shape[0] > 10:
                    if target_arr.size == 0:
                        target_arr = tmp_target_array
                    else:
                        target_arr = np.concatenate([target_arr, tmp_target_array], axis=0)

                print(entire_index+1, ' ', local_index+1)

    if type(from_dirs) == str:
        for entire_index, file in enumerate(os.listdir(from_dirs)):

            tmp_target_array = np.load(from_dirs+file)

            if np.ndim(tmp_target_array) == 2 and tmp_target_array.shape[0] > 10:
                if target_arr.size == 0:
                    target_arr = tmp_target_array
                else:
                    target_arr = np.concatenate([target_arr, tmp_target_array], axis=0)

    print(target_arr.shape)

    mean_1d = np.mean(target_arr, axis=0)
    mean_2d = np.matlib.repmat(mean_1d, target_arr.shape[0], 1)

    std_1d = np.std(target_arr, axis=0)
    tmp_std_min = np.sort(std_1d)

    std_min_nonzero = None
    for min_itr in range(len(tmp_std_min)):
        if tmp_std_min[min_itr] > 0:
            std_min_nonzero = tmp_std_min[min_itr]
            break

    print("std_min_nonzero = ", std_min_nonzero)

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


if __name__ == "__main__":

    save_mean_std_max_min(feature_name="768lab",
                          from_dirs=("../src_data/train_features/OMV200_npy_shot_768lab/",
                                     "../src_data/train_features/OMV62of65_npy_shot_768lab/"),
                          to_dir="./output/mean_std_max_min/")

    # save_mean_std_max_min(feature_name='OMV200_46aco',
    #                       from_dirs=('../src_data/train_features/OMV200_npy_shot_46aco/'),
    #                       to_dir='./output/mean_std_max_min/')
    #
    # save_mean_std_max_min(feature_name='OMV200_91aco',
    #                       from_dirs=('../src_data/train_features/OMV200_npy_shot_91aco/'),
    #                       to_dir='./output/mean_std_max_min/')