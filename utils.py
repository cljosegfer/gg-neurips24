
from scipy import io

def read_ds(dataset, fold_n, dir_path):
    # read
    filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
    dir_path, dataset, fold_n + 1)
    data_mat = io.loadmat(filename)

    # train / test
    train = data_mat['data']['train'][0][0]
    classTrain = data_mat['data']['classTrain'][0][0].ravel()
    test = data_mat['data']['test'][0][0]
    classTest = data_mat['data']['classTest'][0][0].ravel()

    return train, classTrain, test, classTest