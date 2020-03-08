import scipy.io as sio
import numpy as np


def sample(input_path, output_path, is_test=False):
    train_data = sio.loadmat(input_path)

    new_data = []
    new_data_y = []

    new_data_1 = []
    new_data_y_1 = []

    for i in range(10):
        label_id = i + 1
        ori_index = np.array(np.where(train_data['y'] == label_id)[0])
        np.random.shuffle(ori_index)
        index = ori_index[:60]
        label_data = np.array(train_data['X'][:, :, :, index])
        new_data.append(label_data)
        new_data_y.append(np.array(train_data['y'][index, :]))
        if is_test:
            index = ori_index[60: 120]
            label_data = np.array(train_data['X'][:, :, :, index])
            new_data_1.append(label_data)
            new_data_y_1.append(np.array(train_data['y'][index, :]))
    new_data = np.concatenate(new_data, 3)
    new_data_y = np.concatenate(new_data_y, 0)
    sio.savemat(open(output_path, 'wb'), {'X': new_data, 'y': new_data_y}, )
    if is_test:
        new_data = np.concatenate(new_data_1, 3)
        new_data_y = np.concatenate(new_data_y_1, 0)
        print(new_data.shape)
        sio.savemat(open('/tmp/test_small_32x32.mat', 'wb'), {'X': new_data, 'y': new_data_y}, )

sample('train_32x32.mat', '/tmp/train_small_32x32.mat')
sample('test_32x32.mat', '/tmp/val_small_32x32.mat', is_test=True)
