from datasets_process.datasets_custom import dataset, dataset_for_mmd
import torchvision.transforms as transforms
import torch

input_size = 256
transform_imagenet = transforms.Compose([
    transforms.Resize([input_size, input_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class KNDG(object):

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_prepare(self, test=False):
        train_dataset = dataset.MyDataSet(dataset_type='label/KNDG_label', label_type='kndg_shuffle_train',
                                          transform=transform_imagenet)
        val_dataset = dataset.MyDataSet(dataset_type='label/KNDG_label', label_type='kndg_shuffle_test',
                                        transform=transform_imagenet)
        return train_dataset, val_dataset
