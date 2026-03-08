import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from .constants import DATA_SHAPE, NUM_CLASSES, INPUT_CHANNELS

class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: list[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def _get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.detach().clone())

        for module in target_modules:
            module.register_forward_hook(_get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[list[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list
    
class ResNet(DecoupledModel):
    def __init__(self, version, dataset):
        super().__init__()
        self.dataset = dataset
        archs = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        resnet: models.ResNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x):
        if self.dataset in ["mnist", "fmnist", "emnist", "femnist", "usps"]:
            # ResNet expects 3-channel input; replicate channels for grayscale images
            x = x.repeat(1, 3, 1, 1)
        return super().forward(x)

# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    feature_length = {
        "mnist": 1024,
        "medmnistS": 1024,
        "medmnistC": 1024,
        "medmnistA": 1024,
        "covid19": 196736,
        "fmnist": 1024,
        "emnist": 1024,
        "femnist": 1024,
        "cifar10": 1600,
        "cinic10": 1600,
        "cifar100": 1600,
        "tiny_imagenet": 3200,
        "celeba": 133824,
        "svhn": 1600,
        "usps": 800,
    }

    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 512),
                activation3=nn.ReLU(),
            )
        )
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])


# 适用于CIFAR10的轻量CNN模型示例
class SimpleCNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 输入尺寸：3x32x32
            nn.Conv2d(3, 32, 3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x32
            
            nn.Conv2d(32, 64, 3, padding=1), # 16x16x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8x64
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 64, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES[dataset]),  # CIFAR10有10个类别
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        alexnet = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.base = alexnet
        self.classifier = nn.Linear(
            alexnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class LeNet5(DecoupledModel):
    feature_length = {
        "mnist": 256,
        "medmnistS": 256,
        "medmnistC": 256,
        "medmnistA": 256,
        "covid19": 49184,
        "fmnist": 256,
        "emnist": 256,
        "femnist": 256,
        "cifar10": 400,
        "cinic10": 400,
        "svhn": 400,
        "cifar100": 400,
        "celeba": 33456,
        "usps": 200,
        "tiny_imagenet": 2704,
    }

    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
            )
        )

        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])



class MyAlexNet(DecoupledModel):
    """
    used for cifar10
    """
    def __init__(self, dataset: str, use_bn: bool = True):
        super().__init__()
        self.base = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=5, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('avgpool', nn.AdaptiveAvgPool2d((6, 6))),
            ])
        )

        self.classifier = nn.Sequential(
            OrderedDict([
                ('flatten', nn.Flatten(1)),

                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
            
                ('fc3', nn.Linear(4096, NUM_CLASSES[dataset])),
            ])
        )

        if not use_bn:
            for name in list(self.base._modules.keys()):
                if 'bn' in name:
                    del self.base._modules[name]
            for name in list(self.classifier._modules.keys()):
                if 'bn' in name:
                    del self.classifier._modules[name]

class MLP(nn.Module):
    def __init__(self,
                 dataset: str):
        super(MLP, self).__init__()
        infeat = 1
        for _ in range(len(DATA_SHAPE[dataset])):
            infeat *= DATA_SHAPE[dataset][_] 
        self.model = nn.Sequential(
                        nn.Linear(infeat, 200),
                        nn.ReLU(inplace=True),
                        nn.Linear(200, 100),
                        nn.ReLU(inplace=True),
                        nn.Linear(100, NUM_CLASSES[dataset]),
                        # nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # print(f"Input shape: {x.shape}")
        return self.model(x)

class LSTMModel(nn.Module):
    def __init__(self, dataset: str, vocab_size: int = 20000, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0):
    # def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(LSTMModel, self).__init__() # 调用父类的构造方法
        self.num_classes = NUM_CLASSES[dataset]
        self.embedding = nn.Embedding(vocab_size, embed_dim) # vocab_size词汇表大小， embedding_dim词嵌入维度
        self.encoder = nn.LSTM( input_size=embed_dim, 
                                hidden_size=hidden_dim, 
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=bidirectional,
                                dropout=dropout if num_layers > 1 else 0.0
                               )
        self.predictor = nn.Linear(hidden_dim, self.num_classes) # 全连接层
        
    def forward(self, seq):
        output, (hidden, cell) = self.encoder(self.embedding(seq))
        # output :  torch.Size([24, 32, 100])
        # hidden :  torch.Size([1, 32, 100])
        # cell :  torch.Size([1, 32, 100])
        out, _ = torch.max(output, dim=1)
        preds = self.predictor(out) 
        return preds


class RNN(nn.Module):
    def __init__(self, dataset: str, vocab_size = 20000, embed_dim=128, hidden_dim=256, padding_idx=0):
        super(RNN, self).__init__()
        self.num_classes = NUM_CLASSES[dataset]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Simple RNN
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # 分类器
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        # Embedding
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        # RNN
        out, _ = self.rnn(emb)  # (batch, seq_len, hidden_dim)
        out, _ = torch.max(out, dim=1)  # (batch, hidden_dim)
        # 分类
        logits = self.fc(out)
        return logits

class AvgWordEmbClassifier(nn.Module):
    def __init__(self, dataset: str, vocab_size=20000, embed_dim=256):
        super(AvgWordEmbClassifier, self).__init__()
        self.num_classes = NUM_CLASSES[dataset]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 线性分类器 (相当于 Logistic Regression)
        self.fc = nn.Linear(embed_dim, self.num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        emb = self.embedding(x)              # [batch, seq_len, embed_dim]
        avg_emb = emb.mean(dim=1)            # 平均池化 -> [batch, embed_dim]
        logits = self.fc(avg_emb)            # [batch, num_classes]
        return logits

MODELS = {
    "avgcnn": FedAvgCNN,
    "alex": AlexNet,
    "lenet5": LeNet5,
    "myalex": MyAlexNet,
    "simplecnn": SimpleCNN,
    "res18": partial(ResNet, version="18"),
    "res34": partial(ResNet, version="34"),
    "res50": partial(ResNet, version="50"),
    "res101": partial(ResNet, version="101"),
    "res152": partial(ResNet, version="152"),
    "mlp": MLP,
    "lstm": LSTMModel,
    "rnn": RNN,
    "avgwordemb": AvgWordEmbClassifier,
}
