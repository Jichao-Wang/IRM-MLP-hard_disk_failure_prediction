import os
import time
import torch
import pandas as pd
import numpy as np
from torchvision import datasets
from PIL import Image
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def model_to_env(model_name):
    if model_name[:2] == 'HG':
        return 1  # 107
    elif model_name[:2] == 'ST':
        return 2  # 411
    elif model_name[:2] == 'WD':
        return 3  # 8
    elif model_name[:2] == 'TO':
        return 4  # 8
    elif model_name[:2] == 'Sa':
        return 5  # 0
    elif model_name[:2] == 'Hi':
        return 6  # 6
    else:
        return 0


def test_duplicates_broken_disk(file, single_dataset_path, save_image_path):
    # convert file to image with label
    data = pd.read_csv("./mid/data_standard.csv")
    print(data.columns)

    broken_disks = data[1 == data['failure']]["serial_number"]
    broken_disks.to_csv("./mid/broken_disks.csv", index=False)
    print("broken ", len(broken_disks), " disks", len(broken_disks) - len(broken_disks.drop_duplicates()),
          "multiple broken")


def static_all_hard_disks(source_dataset_path, source_dataset_files, static_dataset_path):
    # static all disks serial_number
    # serial_number
    # string

    all_hard_disks = pd.DataFrame(columns=['serial_number'])
    all_records_number = 0

    for i_csv in range(len(source_dataset_files) - 1, -1, -1):
        # len(source_dataset_files)   3
        print("handling csv", source_dataset_files[i_csv])

        mid_csv = pd.read_csv(source_dataset_path + '/' + source_dataset_files[i_csv])
        all_records_number += len(mid_csv)
        mid_serial_list = mid_csv.loc[
            mid_csv['serial_number'] == mid_csv['serial_number'].unique(), ['serial_number', 'model']]

        mid_serial_list.columns = ['serial_number', 'env']
        all_hard_disks = pd.concat([all_hard_disks, mid_serial_list]).drop_duplicates()

        del mid_csv
        del mid_serial_list

    all_hard_disks['env'] = all_hard_disks['env'].apply(model_to_env)
    all_hard_disks.to_csv(static_dataset_path + "/all_hard_disks.csv", index=False)

    print("Create static data at {1}. Totally use {0} log records. Find {2} disks.".format(all_records_number,
                                                                                           time.asctime(time.localtime(
                                                                                               time.time())),
                                                                                           len(all_hard_disks)),
          file=open(static_dataset_path + "/all_disk_name.txt", 'a'))
    return all_hard_disks


def get_all_single_disk_2(source_dataset_path, source_dataset_files, single_dataset_path):
    # could handle limited data size

    # just load some features
    features = [1, 2, 3, 4, 5, 7, 8, 9, 12, 183, 184, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200,
                240, 241, 242]
    columns_specified, num_specified = [], []
    for feature in features:
        columns_specified += ["smart_{0}_raw".format(feature)]
        num_specified += ["smart_{0}_raw".format(feature)]
    num_specified = ["capacity_bytes"] + columns_specified
    columns_specified = ["serial_number", "date", "model", "capacity_bytes", "failure"] + columns_specified

    data = pd.DataFrame(columns=columns_specified)
    for i_csv in range(len(source_dataset_files)):
        print("loading", source_dataset_files[i_csv])
        mid_csv = pd.read_csv(source_dataset_path + '/' + source_dataset_files[i_csv], usecols=columns_specified)
        data = pd.concat([data, mid_csv], axis=0)
        del mid_csv
    # print("sorting")
    data.sort_values(by=['serial_number'], inplace=True)
    data.to_csv("./mid/data.csv", index=False)
    # print("standardization")
    max_min_scaler = lambda x: (x - np.min(x)) * 255 / (np.max(x) - np.min(x))
    data[num_specified] = data[num_specified].apply(max_min_scaler)
    data.to_csv("./mid/data_standard.csv", index=False)

    # print("create broken disks")
    data = pd.read_csv("./mid/data_standard.csv")
    print("total ", len(data), "records")
    broken_disks = data.loc[1 == data['failure'], ["serial_number", "model"]]
    broken_disks['model'] = broken_disks['model'].apply(model_to_env)
    broken_disks.columns = ['serial_number', 'env']
    broken_disks.to_csv("./mid/broken_disks.csv", index=False)
    print("broken ", len(broken_disks), " disks", len(broken_disks) - len(broken_disks.drop_duplicates()),
          "multiple broken")

    # print("grouping")
    data = data.groupby(data[u'serial_number'])
    # print("saving")
    for group in data:
        mid_group = group[1].drop_duplicates()
        mid_group.sort_values("date", inplace=True)
        mid_group.to_csv(single_dataset_path + "/" + str(group[0]) + '.csv', index=False)
        del mid_group


def transform_one_disk(serial_number, failure_flag):
    # transform operations reference to https://blog.csdn.net/BF02jgtRS00XKtCx/article/details/108480177
    label = torch.full([], -1)
    disk = pd.read_csv("./single_disk/" + serial_number + ".csv").drop(
        labels=["serial_number", "date", "model", "failure"],
        axis=1, inplace=False)
    if len(disk) > 28:
        disk = disk[0: 28]
    else:
        return None, None  # records of a disk are less than 28, can not be a picture

    im = Image.fromarray(disk.values.astype(np.uint8))
    # im.save("image/" + serial_number + ".png")
    disk = torch.from_numpy(disk.values.astype(np.uint8))  # transform features to picture matrix
    # print(disk.shape, disk.dtype, disk)

    label = torch.full([], failure_flag)
    # if 0 == failure_flag:
    #     label = torch.full([], 0)  # good disk label
    # elif 1 == failure_flag:
    #     label = torch.full([], 1)  # broken disk label

    return disk, label


def create_my_mnist(sample_number):
    # sample_number is the the number of good disks.
    # broken disks will be used entirely default.
    # the output of this function could be instead of mnist.data and mnist.targets

    # generally, mnist dataset is like the following
    # # mnist = datasets.MNIST('./mnist', train=True, download=True)
    # # mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    # # mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    # references:
    # https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py
    # https://blog.csdn.net/YF_Li123/article/details/76731697

    broken_disks = pd.read_csv("./mid/broken_disks.csv")
    all_disks = pd.read_csv(static_dataset_path + "/all_hard_disks.csv")

    good_disks = all_disks
    good_disks = good_disks.append(broken_disks)
    good_disks = good_disks.append(broken_disks)
    good_disks = good_disks.drop_duplicates(keep=False)

    wjc_data, wjc_target, this_dataset_serial_numbers = [], [], []  # this_dataset: serial number list in this dataset
    global all_envs

    for i in range(len(broken_disks)):
        # get broken hard disks
        disk_image, label = transform_one_disk(broken_disks.iloc[i, 0], 1)
        if disk_image is None:
            continue
        wjc_data.append(disk_image)
        wjc_target.append(label)
        this_dataset_serial_numbers.append(broken_disks.iloc[i, 0])

    number_broken_disks = len(wjc_target)

    # get good hard disks randomly
    # random.seed(121)
    # x = random.sample(range(1, len(good_disks)), sample_number - number_broken_disks)
    x = []
    env_names = broken_disks['env'].unique()
    for i in range(len(env_names)):
        # 全部坏盘
        env_broken = broken_disks.loc[i == broken_disks['env'], :]
        x += env_broken['serial_number'].tolist()
        # 全部硬盘
        env_good_disks = good_disks.loc[i == good_disks['env'], :]
        env_good_disks = env_good_disks['serial_number'].tolist()

        # don`t have enough data. may lack good disks or bad disks or all disks<sample number.
        if len(env_good_disks) + len(env_broken) <= 30:
            # print('env', i, 'don`t have enough disks.')
            continue  # if an env`s all disks less than 30, it will be dropped.

        if len(env_good_disks) >= int(sample_number / len(env_names) - len(env_broken)) and int(
                sample_number / len(env_names) - len(env_broken)) > 0:
            x += random.sample(env_good_disks, int(sample_number / len(env_names) - len(env_broken)))
            # print('env', i, 'add good disks.')
            # all_envs.append(i)
        elif i == 2:
            x += random.sample(env_good_disks, int(len(env_broken) / 3))
            # print('env', i, 'add good disks by special way.')
            # all_envs.append(i)
        # else:
        # print('env', i, 'don`t add good disks.')
        # all_envs.append(i)
        # print('good_disks:', len(env_good_disks))
        # print('broken_disks:', len(env_broken))
        # print('need:', int(sample_number / len(env_names)))

    for serial_number in x:
        disk_image, label = transform_one_disk(serial_number, 0)
        if disk_image is None:
            continue
        wjc_data.append(disk_image)
        wjc_target.append(label)
        this_dataset_serial_numbers.append(serial_number)
    number_good_disks = len(wjc_target) - number_broken_disks
    print(len(wjc_target), " disks, ", number_broken_disks, " broken", number_good_disks, " good")

    # shuffle data set
    c = list(zip(wjc_data, wjc_target, this_dataset_serial_numbers))
    random.shuffle(c)
    wjc_data, wjc_target, this_dataset_serial_numbers = zip(*c)
    # print(wjc_target)

    wjc_data = torch.stack(wjc_data, dim=0)
    wjc_target = torch.stack(wjc_target, dim=0)

    # merge data and target together
    # mnist_like_data_set = []
    # for i in range(len(wjc_data)):
    #     mid = (wjc_data[i], wjc_target[i])
    #     mnist_like_data_set.append(mid)

    return wjc_data, wjc_target, this_dataset_serial_numbers


def Kmeans_env_tag(wjc_data, this_dataset_serial_numbers, k, return_km_labels_flag=False):
    # 2 methods of environments dividing
    # method 1: env is related to model. Which has been realized by default env tag generated by model.
    # method 2: env is calculated by k-means cluster. Here, do it and rewrite all_hard_disks.csv and broken_disks.csv.

    # Note: This method will change env column in csv files. Please use it careful.

    mid_wjc_data = torch.reshape(wjc_data, (
        wjc_data.shape[0], 1, wjc_data.shape[1] * wjc_data.shape[2])).squeeze()  # reshape wjc_data from 3d to 2d
    km = KMeans(n_clusters=k, random_state=6002).fit(mid_wjc_data)

    if return_km_labels_flag:
        return list(range(k)), km.labels_
    else:
        all_hard_disks = pd.read_csv(static_dataset_path + "/all_hard_disks.csv")
        broken_disks = pd.read_csv("./mid/broken_disks.csv")
        for i in range(len(km.labels_)):
            if this_dataset_serial_numbers[i] in broken_disks['serial_number'].values:
                broken_disks.loc[broken_disks['serial_number'] == this_dataset_serial_numbers[i], 'env'] = km.labels_[i]
            if this_dataset_serial_numbers[i] in all_hard_disks['serial_number'].values:
                all_hard_disks.loc[all_hard_disks['serial_number'] == this_dataset_serial_numbers[i], 'env'] = \
                    km.labels_[i]
        all_hard_disks.to_csv(static_dataset_path + "/all_hard_disks.csv", index=False)
        broken_disks.to_csv("./mid/broken_disks.csv", index=False)
        return list(range(k))


def test_k_Kmeans_env_tag(wjc_data):
    # Find best k for k-means cluster.
    global k
    global all_envs
    mid_wjc_data = torch.reshape(wjc_data, (
        wjc_data.shape[0], 1, wjc_data.shape[1] * wjc_data.shape[2])).squeeze()  # reshape wjc_data from 3d to 2d

    silhouetteScore = []
    for i in range(3, 16):
        km = KMeans(n_clusters=i, random_state=6002).fit(mid_wjc_data)
        score = silhouette_score(mid_wjc_data, km.labels_)
        silhouetteScore.append(score)
    # print(silhouetteScore)
    plt.figure(figsize=(10, 6))
    plt.plot(range(3, 16), silhouetteScore, linewidth=1.5, linestyle="-")
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.savefig(static_dataset_path + '/best k.png')
    plt.show()
    k = silhouetteScore.index(max(silhouetteScore)) + 3
    all_envs = list(range(k))
    # print(silhouetteScore)
    # print(silhouetteScore.index(max(silhouetteScore)), max(silhouetteScore))
    return k


def setup_seed(seed):
    # 设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(100)
source_dataset_path = "backblaze_2019"
static_dataset_path = "statics_results"
single_dataset_path = "single_disk"
save_image_path = "image"
k = -1
all_envs = []

if __name__ == '__main__':
    start = time.time()
    source_dataset_files = os.listdir(source_dataset_path)[:300]  # only use 100 days data

    # step 1 create all disk name
    # all_hard_disks = static_all_hard_disks(source_dataset_path, source_dataset_files, static_dataset_path)
    all_hard_disks = pd.read_csv(static_dataset_path + "/all_hard_disks.csv")
    print("totally", len(all_hard_disks), "disks")  # totally 307719 disks, 9944597 records, 359 failure

    end = time.time()
    print("step 1 totally cost ", end - start, " seconds")
    start = time.time()

    # step 2 create single data
    # get_all_single_disk_1(all_hard_disks, source_dataset_path, source_dataset_files, single_dataset_path)
    # get_all_single_disk_2(source_dataset_path, source_dataset_files, single_dataset_path)

    end = time.time()
    print("step 2 totally cost ", end - start, " seconds")
    start = time.time()

    # step 3 transform single data to mnist format
    wjc_data, wjc_target, this_dataset_serial_numbers = create_my_mnist(1000)

    end = time.time()
    print("step 3 totally cost ", end - start, " seconds")
    start = time.time()

    # step 4 k-means env tag
    test_k_Kmeans_env_tag(wjc_data)
    all_envs = Kmeans_env_tag(wjc_data, this_dataset_serial_numbers, k)

    end = time.time()
    print("step 4 totally cost ", end - start, " seconds")
