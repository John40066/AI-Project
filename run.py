import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import cv2


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # using resnet18
        self.resnet = torchvision.models.resnet18(pretrained=False)
        # over-write the first conv layer
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),)

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate features of 2 images
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)

        return output


class APP_MATCHER(Dataset):
    def __init__(self, trainData):
        super(APP_MATCHER, self).__init__()
        self.data = trainData

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # get the 2 images
        image_1 = np.asarray(self.data[index][0], dtype=np.float32).copy()
        image_1 = torch.from_numpy(np.moveaxis(image_1, -1, 0))
        image_2 = np.asarray(self.data[index][1], dtype=np.float32).copy()
        image_2 = torch.from_numpy(np.moveaxis(image_2, -1, 0))

        return image_1, image_2, self.data[index][2], self.data[index][3]


def HistCheck(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    mx, total = 0, 0
    for i in range(len(hist)):
        if hist[i] > mx:
            mx = hist[i]
        total += hist[i]
    if(mx / total >= 0.95):
        return True
    return False


def Get_Images(fp1, fp2, sz=100):
    img_list = []
    h_sz = sz // 2

    src1 = cv2.imread(fp1)
    src2 = cv2.imread(fp2)

    h, w = len(src1), len(src1[0])

    for i in range(h//h_sz - 2):
        for j in range(w//h_sz - 2):
            y, x = i * h_sz, j*h_sz
            c_src1 = src1[y:y+sz, x:x+sz]

            if HistCheck(c_src1):
                continue

            l, r = max(x - h_sz, 0), min(x + sz+h_sz, w)
            u, d = max(y - h_sz, 0), min(y + sz+h_sz, h)
            c_src2 = src2[u:d, l:r]
            tmp = cv2.matchTemplate(
                c_src1, c_src2, cv2.TM_CCOEFF_NORMED)
            _, _, _, pt = cv2.minMaxLoc(tmp)
            c_src2 = c_src2[pt[1]:pt[1]+sz, pt[0]:pt[0]+sz]

            c_src1 = cv2.cvtColor(c_src1, cv2.COLOR_RGB2BGR)
            c_src2 = cv2.cvtColor(c_src2, cv2.COLOR_RGB2BGR)

            img_list.append((c_src1, c_src2, y, x))

    return img_list


PATH = './Data/SiamsesModel_v2.pt'
imgSrc1 = './Data/test_img1.png'
imgSrc2 = './Data/test_img2.png'


model = SiameseNetwork()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

img_list = Get_Images(imgSrc1, imgSrc2)


test_dataset = APP_MATCHER(img_list)
t_kwargs = {'batch_size': 1}
t_loader = torch.utils.data.DataLoader(test_dataset, **t_kwargs)


with torch.no_grad():
    i = 0
    Xparts = []
    for i, images in tqdm(enumerate(t_loader)):
        # predict
        outputs = model(images[0], images[1]).squeeze()
        if outputs <= 0.5:
            Xparts.append((images[2], images[3]))

    img1 = cv2.imread(imgSrc1)
    img2 = cv2.imread(imgSrc2)
    img1_c = img1.copy()

    for i in range(len(Xparts)):
        b = Xparts[i][0][0].item()
        a = Xparts[i][1][0].item()
        point1 = (a, b)
        point2 = (a + 100, b + 100)
        cv2.rectangle(img1_c, point1, point2, [255, 0, 255], -1)

    img1_c = cv2.resize(img1_c, [img1_c.shape[1]//2, img1_c.shape[0]//2])
    img1 = cv2.resize(img1, [img1.shape[1]//2, img1.shape[0]//2])
    img2 = cv2.resize(img2, [img2.shape[1]//2, img2.shape[0]//2])
    combine = cv2.addWeighted(img1_c, 0.2, img1, 0.8, 0)
    cv2.imshow('winname', combine)
    cv2.imshow('winname2', img2)
    cv2.waitKey(0)
