from torch import nn
import torch
import torch.nn.functional as F


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def __call__(self, tensor):
        x = tensor.clone()
        for l in range(list(x.size())[0]):
            line = x[l].clone()
            line=line/((line.pow(2).sum()).pow(0.5))
            x[l]=line
        return x

#1st Autoencoder
class Encoder(nn.Module):
    def __init__(self, bands_nb, patch_size):
        input_size = (bands_nb, patch_size, patch_size)

        super(Encoder, self).__init__()

        p = 0.2
        # Stage 1
        # Feature extraction
        self.conv11 = nn.Conv2d(bands_nb, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        # self.dropout11 = nn.Dropout2d(p=p)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        # self.dropout12 = nn.Dropout2d(p=p)
        self.conv13 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(32)
        # self.dropout13 = nn.Dropout2d(p=p)
        self.conv14 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(32)
        # self.dropout14 = nn.Dropout2d(p=p)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()
        self.l2norm = L2()


    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x13 = F.relu(self.bn13(self.conv13(x12)))
        x14 = self.conv14(x13)
        #x14 = F.relu(x13)

        # x11 = F.relu(self.bn11(self.conv11(x)))
        # x14 = (self.bn13(self.conv13(x11)))
        # x14 = self.conv14(x13)

        size14 = x14.size()
        #pourquoi ? .view
        x14_ = x14.view(size14[0], size14[1], size14[2]*size14[3])
        x14_ = F.normalize(x14_, p=2, dim=2) #l2 norm
        encoded = x14_.view(size14[0], size14[1], size14[2], size14[3])


        return encoded


class Decoder(nn.Module):
    def __init__(self, bands_nb, patch_size):
        input_size = (bands_nb, patch_size, patch_size)

        super(Decoder, self).__init__()

        p=0.2
        # Stage 1d
        # Feature decoder
        self.conv14d = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn14d = nn.BatchNorm2d(32)
        # self.dropout14d = nn.Dropout2d(p=p)
        self.conv13d = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn13d = nn.BatchNorm2d(16)
        # self.dropout13d = nn.Dropout2d(p=p)
        self.conv12d = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        # self.dropout12d = nn.Dropout2d(p=p)
        self.conv11d = nn.Conv2d(16, bands_nb, kernel_size=3, padding=1)
        # self.dropout11d = nn.Dropout2d(p=p)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()


    def forward(self, x):
        # Stage 1d
        x14d = F.relu(self.bn14d(self.conv14d(x)))
        x13d = F.relu(self.bn13d(self.conv13d(x14d)))
        x12d = F.relu(self.bn12d(self.conv12d(x13d)))
        x11d = torch.sigmoid(self.conv11d(x12d))

        # x13d = F.relu(self.bn13d(self.conv13d(x)))
        # x11d = torch.sigmoid(self.conv11d(x13d))

        decoded = x11d

        return decoded


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        e, id1 = self.ecoder(x)
        d = self.decoder(e, id1)
        return e, d