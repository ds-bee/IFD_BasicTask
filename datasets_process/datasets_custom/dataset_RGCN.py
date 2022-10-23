import os

import torch
from sklearn.model_selection import train_test_split

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

from data_GCN.PathGraph import PathGraph

transform = transforms.Compose(
        [transforms.Resize([64, 64]),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])       # jiancha huachulaide tupian


class MyDataSet_RGCN(object):
    
    def __init__(self, dataset_type, label_type):
        dataset_path = ''
        # self.transform = transform
        self.sample_list = list()
        self.dataset_type = dataset_type
        f = open(dataset_path + self.dataset_type + '/'+label_type+'.txt')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()

    def __getitem__(self, index):
        item = self.sample_list[index]
        img = plt.imread(item.split(' ', 1)[0])
        # img = Image.open(item.split(' ', 1)[0]).convert('RGB')                  # 标签文件使用的‘ ’做分隔符；
        # img = Image.open(item.split(' ', 1)[0])              # 标签文件使用的‘ ’做分隔符；
        # if self.transform is not None:
        #     img = self.transform(img)

        # img = np.array(img)
        # img = image.reshape((1,64, 64))
        label = int(item.split(' ')[-1])                # -1是标签
        img = torch.from_numpy(img)
        data = dataGCN_load(img,label,"Node")

        return data

    def __len__(self):
        return len(self.sample_list)

    # def data_preprare(self, test=False):
    #     if len(os.path.basename(self.data_dir).split('.')) == 2:
    #         with open(self.data_dir, 'rb') as fo:
    #             list_data = pickle.load(fo, encoding='bytes')
    #     else:
    #         list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
    #         with open(os.path.join(self.data_dir, "CWRUPath.pkl"), 'wb') as fo:
    #             pickle.dump(list_data, fo)
    #
    #     if test:
    #         test_dataset = list_data
    #         return test_dataset
    #     else:
    #
    #         train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
    #
    #         return train_dataset, val_dataset
def dataGCN_load(img,label,task):
    # print(img.size())
    data = []
    x = img[0:32,0:32]
    x = torch.reshape(x, (1, 32 * 32))
    x = torch.squeeze(x, dim=0)
    data.append(x)
    x = img[0:32,32:64]
    x = torch.reshape(x, (1, 32 * 32))
    x = torch.squeeze(x, dim=0)
    data.append(x)
    x = img[32:64,0:32]
    x = torch.reshape(x, (1, 32 * 32))
    x = torch.squeeze(x, dim=0)
    data.append(x)
    x = img[32:64,32:64]
    x = torch.reshape(x, (1, 32 * 32))
    x = torch.squeeze(x, dim=0)
    data.append(x)
    graphset = PathGraph(4,data,label,task)

    return graphset






if __name__ == '__main__':

    # ds = MyDataSet('/45TB/hyq/lyb/IFD_BasicTask/label/CWSU_label', 'png_0_cwsu_train')
    # print('样本数：', ds.__len__())
    # img, gt = ds.__getitem__(4)
    # # print('img 类型：', type(img))
    # print('img 标签：', gt)
    # print('img 维度：', img.size())
    # # imshow(img)
    # loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    # for data in loader:
    #     img, label = data
    # print(type(img))
    # print(img.shape)
    # print(type(label))R
    # print(label.shape)

    ds = MyDataSet_RGCN('D:\IFD_BasicTask\label\CWRU_GCN_lable', 'array_0_cwru_train')


    # print('img 类型：', type(img))
    # print('img 标签：', gt)
    # print("s")

    # print('img 维度：', img.size())
    # imshow(img)
    from torch_geometric.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    for data in loader:
        data = data
        las = data.y
    # print(type(img))
    # print(img.shape)
    # print(type(label))
    # print(label.shape)


