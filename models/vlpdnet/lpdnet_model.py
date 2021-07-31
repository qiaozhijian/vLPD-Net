import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.point_utils import get_graph_feature_Origin, get_graph_feature, knn


class LPDNetOrign(nn.Module):
    def __init__(self, emb_dims=512, channels=[64, 64, 128]):
        super(LPDNetOrign, self).__init__()
        self.act_f = nn.LeakyReLU(inplace=True, negative_slope=1e-2)
        self.k = 20
        self.emb_dims = emb_dims
        self.convDG1 = nn.Sequential(nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels[0]), self.act_f)
        self.convDG2 = nn.Sequential(nn.Conv2d(channels[0], channels[0], kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels[0]), self.act_f)
        self.convSN1 = nn.Sequential(nn.Conv2d(channels[0], channels[0], kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels[0]), self.act_f)
        self.convSN2 = nn.Sequential(nn.Conv2d(channels[0], channels[0], kernel_size=1, bias=False),
                                     nn.BatchNorm2d(channels[0]), self.act_f)
        self.conv1_lpd = nn.Sequential(nn.Conv1d(3, channels[0], kernel_size=1, bias=False),
                                       nn.BatchNorm1d(channels[0]), self.act_f)
        self.conv2_lpd = nn.Sequential(nn.Conv1d(channels[0], channels[0], kernel_size=1, bias=False),
                                       nn.BatchNorm1d(channels[0]), self.act_f)
        self.conv3_lpd = nn.Sequential(nn.Conv1d(channels[0], channels[1], kernel_size=1, bias=False),
                                       nn.BatchNorm1d(channels[1]), self.act_f)
        self.conv4_lpd = nn.Sequential(nn.Conv1d(channels[1], channels[2], kernel_size=1, bias=False),
                                       nn.BatchNorm1d(channels[2]), self.act_f)
        self.conv5_lpd = nn.Sequential(nn.Conv1d(channels[2], self.emb_dims, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(self.emb_dims), self.act_f)

    # input x: # [B,1,num,num_dims]
    # output x: # [b,emb_dims,num,1]
    def forward(self, x):
        x = torch.squeeze(x, dim=1).transpose(2, 1)  # [B,num_dims,num]
        batch_size, num_dims, num_points = x.size()

        xInit3d = x

        x = self.conv1_lpd(x)
        x = self.conv2_lpd(x)

        # Serial structure
        # Danymic Graph cnn for feature space
        x = get_graph_feature_Origin(x, k=self.k)  # [b,channel*2,num,20]
        x = self.convDG1(x)  # [b,channel,num,20]
        x = self.convDG2(x)  # [b,channel,num,20]
        x = x.max(dim=-1, keepdim=True)[0]  # [b,channel,num,1]

        # Spatial Neighborhood fusion for cartesian space
        idx = knn(xInit3d, k=self.k)
        x = get_graph_feature_Origin(x, idx=idx, k=self.k, cat=False)  # [b,channel,num,20]
        x = self.convSN1(x)  # [b,channel,num,20]
        x = self.convSN2(x)  # [b,channel,num,20]
        x = x.max(dim=-1, keepdim=True)[0].squeeze(-1)  # [b,channel,num]

        x = self.conv3_lpd(x)  # [b,channel,num]
        x = self.conv4_lpd(x)  # [b,128,num]
        x = self.conv5_lpd(x)  # [b,emb_dims,num]
        x = x.unsqueeze(-1)  # [b,emb_dims,num,1]

        return x


class LPDNet(nn.Module):
    """Implement for LPDNet using pytorch package.

    """

    def __init__(self, emb_dims=1024, channels=[64, 128, 256]):
        super(LPDNet, self).__init__()
        self.negative_slope = 0.0
        self.k = 20
        self.emb_dims = emb_dims
        # [b,6,num,20]
        self.convDG1 = nn.Sequential(nn.Conv2d(channels[0] * 2, channels[1], kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope=self.negative_slope))
        self.convDG2 = nn.Sequential(nn.Conv2d(channels[1], channels[1], kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope=self.negative_slope))
        self.convSN1 = nn.Sequential(nn.Conv2d(channels[1] * 2, channels[2], kernel_size=1, bias=True),
                                     nn.LeakyReLU(negative_slope=self.negative_slope))

        self.conv1_lpd = nn.Conv1d(3, channels[0], kernel_size=1, bias=True)
        self.conv2_lpd = nn.Conv1d(channels[0], channels[0], kernel_size=1, bias=True)
        self.conv3_lpd = nn.Conv1d((channels[1] + channels[1] + channels[2]), self.emb_dims, kernel_size=1, bias=True)

    # input x: # [B,num_dims,num]
    # output x: # [b,emb_dims,num]
    def forward(self, x):
        x = torch.squeeze(x, dim=1).transpose(2, 1)  # [B,num_dims,num]
        batch_size, num_dims, num_points = x.size()
        #
        xInit3d = x

        x = F.leaky_relu(self.conv1_lpd(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.conv2_lpd(x), negative_slope=self.negative_slope)

        # Serial structure
        # Dynamic Graph cnn for feature space
        x = get_graph_feature(x, k=self.k)  # [b,64*2,num,20]
        x = self.convDG1(x)  # [b,128,num,20]
        x1 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]
        x = self.convDG2(x)  # [b,128,num,20]
        x2 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]

        # Spatial Neighborhood fusion for cartesian space
        idx = knn(xInit3d, k=self.k)
        x = get_graph_feature(x2.squeeze(-1), idx=idx, k=self.k)  # [b,128*2,num,20]
        x = self.convSN1(x)  # [b,256,num,20]
        x3 = x.max(dim=-1, keepdim=True)[0]  # [b,256,num,1]

        x = torch.cat((x1, x2, x3), dim=1).squeeze(-1)  # [b,512,num]
        x = F.leaky_relu(self.conv3_lpd(x), negative_slope=self.negative_slope).view(batch_size, -1,
                                                                                     num_points)  # [b,emb_dims,num]
        x = x.unsqueeze(-1)  # [b,emb_dims,num,1]
        return x
