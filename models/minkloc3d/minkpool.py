# Author: Jacek Komorowski
# Warsaw University of Technology

import MinkowskiEngine as ME
import torch

import models.minkloc3d.pooling as pooling
from models.minkloc3d.minkfpn import MinkFPN


class MinkPool(torch.nn.Module):
    def __init__(self, model_params, in_channels):
        super().__init__()
        self.model = model_params.mink_model
        self.in_channels = in_channels
        self.feature_size = model_params.feature_size  # Size of local features produced by local feature extraction block
        self.output_dim = model_params.output_dim  # Dimensionality of the global descriptor
        self.backbone = MinkFPN(in_channels=in_channels, model_params=model_params)
        self.n_backbone_features = model_params.output_dim
        self.pooling = pooling.GeM()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [B,dims,num_points,1], [B,1,num_points,3]
    def __input(self, feat_map, coords_xyz=None):
        # coords_xyz = coords_xyz.cpu().squeeze(1)
        coords = []
        feat_maps = []
        for e in range(len(coords_xyz)):
            coords_e, inds_e = ME.utils.sparse_quantize(coords_xyz[e][0].cpu().squeeze(1), return_index=True,
                                                        quantization_size=0.01)
            feat_map_e = feat_map[e][:, inds_e, :].squeeze(-1).transpose(-1, -2)
            coords.append(coords_e)
            feat_maps.append(feat_map_e)
        # coords = [ME.utils.sparse_quantize(coords_xyz[e,...], quantization_size=0.01)
        #           for e in range(coords_xyz.shape[0])]
        coords = ME.utils.batched_coordinates(coords)
        # Assign a dummy feature equal to 1 to each point
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        feats = torch.vstack(feat_maps)
        batch = {'coords': coords, 'features': feats}
        return batch

    def forward(self, feat_map, coords_xyz=None):
        batch = self.__input(feat_map, coords_xyz=coords_xyz)
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'], device=self.device)
        x = self.backbone(x)
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(
            x.shape[1], self.feature_size)
        x = self.pooling(x)
        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1],
                                                                                                    self.output_dim)
        # x is (batch_size, output_dim) tensor
        return {'g_fea': x}

    def print_info(self):
        print('Model class: MinkPool')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Aggregation parameters: {}'.format(n_params))
        if hasattr(self.backbone, 'print_info'):
            self.backbone.print_info()
        if hasattr(self.pooling, 'print_info'):
            self.pooling.print_info()
