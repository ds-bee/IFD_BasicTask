from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])       # jiancha huachulaide tupian


class mmdDataSet(Dataset):
    def __init__(self, dataset_type1, dataset_type2, label_type1, label_type2, transform_src,
                 transform_tgt, update_dataset=False):
        dataset_path = './'
        self.transform = [transform_src, transform_tgt]
        self.sample_list = list()
        self.sample_list2 = list()
        self.dataset_type = [dataset_type1, dataset_type2]
        f = open(dataset_path + self.dataset_type[0] + '/'+label_type1+'.txt')
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()
        f2 = open(dataset_path + self.dataset_type[-1] + '/'+label_type2+'.txt')
        lines = f2.readlines()
        for line in lines:
            self.sample_list2.append(line.strip())
        f2.close()

    def __getitem__(self, index):
        item = self.sample_list[index]
        item2 = self.sample_list2[index]
        img = Image.open(item.split(' ', 1)[0]).convert('RGB')  # PIL图像 #标签文件使用的‘ ’做分隔符；
        img2 = Image.open(item2.split(' ', 1)[0]).convert('RGB')
        if self.transform is not None:
            img = self.transform[0](img)
            img2 = self.transform[1](img2)
        label = int(item.split(' ')[-1])                # -1是标签，源域才有标签；
        label2 = int(item2.split(' ')[-1])
        return [img, img2], [label, label2]

    def __len__(self):
        return len(self.sample_list)   # 一定是较短的


def main():
    
    print('hello world')
    ds = mmdDataSet('gear_labels', 'train_label_1')
    print('样本数：', ds.__len__())
    img, gt = ds.__getitem__(4)
    # print('img 类型：', type(img))
    print('img 标签：', gt)
    print('img 维度：', img.size())
    # imshow(img)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    for data in loader:
        img, label = data
    print(type(img))
    print(img.shape)
    print(type(label))
    print(label.shape)


# 检验编写的类是否正确
if __name__ == '__main__':
    main()




