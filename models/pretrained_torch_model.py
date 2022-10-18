import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class PretrainedTorchModel(nn.Module):
    def __init__(self, num_class, torch_model):
        super().__init__()
        self.model_name = torch_model
        if self.model_name == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=False)
        elif self.model_name == 'vgg11':
            self.model = torchvision.models.vgg11(pretrained=False)
        elif self.model_name == 'densenet121':
            self.model = torchvision.models.densenet121(pretrained=False)
        else:
            raise Exception("Please input the pretrained model_name is prepared!", self.model_name)

    def forward(self, input):
        pred = self.model(input)
        return pred


if __name__ == "__main__":

    import torch
    from torch.autograd import Variable
    x = torch.randn(3, 64, 64)
    net = PretrainedTorchModel()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
    print(x.shape)
    y = net(x)

