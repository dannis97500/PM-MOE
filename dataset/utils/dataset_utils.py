# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.1 # for Dirichlet distribution. 100 for exdir

def check(config_path, train_path, test_path, num_clients, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_cifar_noniid_s(dataset, num_users, noniid_s=20, local_size=600, train=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(2022)
    s = noniid_s / 100
    num_per_user = local_size if train else 300
    num_classes = len(np.unique(dataset.targets))

    noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]

    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_iid = int(num_per_user*s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_users)}
    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [2000 for i in range(num_classes)] if train else [500 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes-1)*iid_per_label

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+iid_num]), axis=0)
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # rand_label = np.random.choice(label_list, 3, replace=False)
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:start+noniid_num]), axis=0)
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)
    return dict_users



def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2)) 

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples: 
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K): # Dirichlet random partitioning is performed on each class
                idx_k = np.where(dataset_label == k)[0] # Find all data indexes of the current category k
                np.random.shuffle(idx_k)  # Randomly shuffle the index of the current category
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) # Use Dirichlet distribution to generate the distribution ratio, alpha is a parameter, num_clients is the number of clients
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)]) # Adjust the ratio based on the current number of samples for each client
                proportions = proportions/proportions.sum() # Renormalize the ratios to ensure they sum to 1
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1] # Compute the split points for the sample indices that each client should receive.
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))] # Assign sample indexes to each client based on the split point, and directly assign the indexes to the idx_batch list
                min_size = min([len(idx_j) for idx_j in idx_batch]) # Update min_size to the minimum number of samples among all clients.
            try_cnt += 1

        for j in range(num_clients): 
            dataidx_map[j] = idx_batch[j]
    
    elif partition == 'exdir':
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client
        
        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        # You can adjust the `min_require_size_per_label` to meet you requirements
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            # initialize
            for k in range(num_classes):
                clientidx_map[k] = []
            # allocate
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
        
        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                # Case 1 (original case in Dir): Balance the number of sample per client
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # process the remainder samples
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    
    elif partition == 's_par':
        print("\n***** s partition start *****")
        noniid_s = 20 # Heterogeneity ratio, 20 means 20% isomorphism
        s = noniid_s/ 100  
      
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples: # Each client has at least 40 samples. If the division requirement is not met, re-divide it.
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K): # Dirichlet random partitioning is performed on each class
                idx_k = np.where(dataset_label == k)[0] # Find all data indexes of the current category k
                np.random.shuffle(idx_k)  # Randomly shuffle the index of the current category
                
                # Traverse the current class index list and evenly select the clients that need to be allocated. It is necessary to consider whether homogeneous samples can be allocated.
                num_iid = int(len(idx_k)*s) # The percentage of the sample size of the current category is used to distribute iid, that is, the total sample size of iid, and the rest are Dirichlet distribution
                iid_per_label = int(num_iid/len(idx_batch))  #  Evenly distribute according to the number of clients
                iid_per_label_last = num_iid - (len(idx_batch)-1) * iid_per_label  # The last client is treated specially, and the part that cannot be divided evenly is given to the last client.
                for i in range(num_clients - 1):
                    idx_batch[i] = idx_batch[i] + idx_k[i*iid_per_label:(i+1)*iid_per_label].tolist()
                    # idx_batch[i].append(idx_k[i*iid_per_label:(i+1)*iid_per_label].tolist())
                idx_batch[num_clients-1] = idx_batch[num_clients-1] + idx_k[(i+1)*iid_per_label : (i+1)*iid_per_label+iid_per_label_last].tolist()
                # Delete the assigned client index list and perform Dirichlet random assignment on the remaining ones
                remaining_idx_k = np.delete(idx_k, np.s_[0:num_iid])
                
                # non_iid allocation
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) # Use Dirichlet distribution to generate the distribution ratio, alpha is a parameter, num_clients is the number of clients
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)]) # Adjust the ratio based on the current number of samples for each client
                proportions = proportions/proportions.sum() # Renormalize the ratios to ensure they sum to 1
                proportions = (np.cumsum(proportions)*len(remaining_idx_k)).astype(int)[:-1] # Compute the split points for the sample indices that each client should receive.
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(remaining_idx_k, proportions))] # Assign sample indexes to each client based on the split point, and directly assign the indexes to the idx_batch list
                min_size = min([len(idx_j) for idx_j in idx_batch]) # Update min_size to the minimum number of samples among all clients.
            try_cnt += 1

        for j in range(num_clients): 
            dataidx_map[j] = idx_batch[j]
        
    else:
        raise NotImplementedError

    # assign data，Allocate data according to the filtered index
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data






def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
