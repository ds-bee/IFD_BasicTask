from datasets_process.datasets_custom import dataset, dataset_for_mmd, dataset_GCN, dataset_RGCN
import torchvision.transforms as transforms
import torch
from torch_geometric.data import DataLoader

input_size = 64
transform_imagenet = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class CWRU_RGCN(object):

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_prepare(self, test=False):
        train_dataset = dataset_RGCN.MyDataSet_RGCN(dataset_type='D:\IFD_BasicTask\label\CWRU_GCN_lable', label_type='array_0_cwru_train'
                                          )
        val_dataset = dataset_RGCN.MyDataSet_RGCN(dataset_type='D:\IFD_BasicTask\label\CWRU_GCN_lable', label_type='array_0_cwru_test'
                                       )
        return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = CWRU_RGCN.data_prepare()
    loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    for data in loader:
        data = data
        label = data.y