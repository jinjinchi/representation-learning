import torch.nn as nn
import torch
import torch.nn.functional as F


def tile_and_concat(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Merge 1D and 2D tensor (use to aggregate feature maps and representation
    and compute local mutual information estimation)

    Args:
        tensor (torch.Tensor): 2D tensor (feature maps)
        vector (torch.Tensor): 1D tensor representation

    Returns:
        torch.Tensor: Merged tensor (2D)
    """

    B, C, H, W = tensor.size()
    vector = vector.unsqueeze(2).unsqueeze(2) 
    expanded_vector = vector.expand((B, vector.size(1), H, W)) 
    local_input = torch.cat([tensor, expanded_vector], dim=1)
    flatten = nn.Flatten()
    x = flatten(local_input)
    return local_input,x
    

def tile_and_concat_one(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Merge 1D and 2D tensor (use to aggregate feature maps and representation
    and compute local mutual information estimation)

    Args:
        tensor (torch.Tensor): 2D tensor (feature maps)
        vector (torch.Tensor): 1D tensor representation

    Returns:
        torch.Tensor: Merged tensor (2D)
    """

    B, C, H, W = tensor.size()
    vector = vector.unsqueeze(2).unsqueeze(2)
    expanded_vector = vector.expand((B, vector.size(1), H, W))
    return torch.cat([tensor, expanded_vector], dim=1)
    

def global_and_concat(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    flattened_tensor = nn.Flatten()(tensor)
    return torch.cat([flattened_tensor, vector], dim=1)
    

def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1) and (
            classname.find('Cond') == -1) and (classname.find('Spectral') == -1):
        try:
            # Normal conv layer
            m.weight.data.normal_(0.0, 0.02)
        except:
            # Conv layer with spectral norm
            m.weight_u.data.normal_(0.0, 0.02)
            m.weight_v.data.normal_(0.0, 0.02)
            m.weight_bar.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1 and classname.find('cond') == -1 and classname.find('Cond') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
class LocalStatisticsNetworkSH(nn.Module):
    def __init__(self, img_feature_channels: int):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=512, kernel_size=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=512)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.conv3 = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1
        )
        print("new local gp three bn")
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01) 

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        local_statistics = self.conv3(x)
        return local_statistics
        
        

class GlobalStatisticsNetworkSH(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): Dimension of the representationss
    """

    def __init__(
            self, feature_map_size: list, feature_map_channels: int, latent_dim: int
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=feature_map_size[0] * feature_map_size[1] * feature_map_channels + latent_dim,
            out_features=512,
        )
        
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        print("new global gp three")
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)  

    def forward(
            self, concat_feature: torch.Tensor
    ) -> torch.Tensor:
        x = self.dense1(concat_feature)
        x = self.leakyrelu(x)
        x = self.dense2(x)
        x = self.leakyrelu(x)
        global_statistics = self.dense3(x)

        return global_statistics
        
        
class LocalStatisticsNetworkEX(nn.Module):
    def __init__(self, img_feature_channels: int):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=512, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1
        )
        print("local EX three no bn")
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01) 

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        local_statistics = self.conv3(x)
        return local_statistics
        
        

class GlobalStatisticsNetworkEX(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): Dimension of the representationss
    """

    def __init__(
            self, feature_map_size: list, feature_map_channels: int, latent_dim: int
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=feature_map_size[0] * feature_map_size[1] * feature_map_channels + latent_dim,
            out_features=512,
        )
        
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        print("global EX three")
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)  

    def forward(
            self, concat_feature: torch.Tensor
    ) -> torch.Tensor:
        x = self.dense1(concat_feature)
        x = self.leakyrelu(x)
        x = self.dense2(x)
        x = self.leakyrelu(x)
        global_statistics = self.dense3(x)

        return global_statistics
        
        
        
class WeightNet(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(WeightNet, self).__init__()
        ndf = 512
        nc = in_channels

        self.network = nn.Sequential(
            nn.Conv2d(nc, ndf, 2, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ndf, ndf, 2, 1, 1, bias=True),
            #nn.BatchNorm2d(num_features=ndf),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ndf, ndf, 2, 1, 1, bias=True),
            #nn.BatchNorm2d(num_features=ndf),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(ndf, out_dim, 2, 1, 1, bias=True)
        )
        self.network.apply(weights_init)

    def forward(self, input, label=None):
        weight_logits = self.network(input)
        weight_logits = torch.sum(weight_logits, (2, 3))
        weight_logits = weight_logits.view(weight_logits.size(0), -1)

        return F.relu(weight_logits)


class Critic(nn.Module):
    def __init__(self, nb_linear=4, input_dim=600):
        super(Critic, self).__init__()

        self.hidden_dim = 768
        #self.hidden_dim = 1024
        self.input_dim = input_dim

        if nb_linear == 2:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 4
                nn.Linear(self.hidden_dim, 1)
            )
        elif nb_linear == 3:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                # 2
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                # 4
                nn.Linear(self.hidden_dim, 1)
            )
        else:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1)
            )

    def forward(self, x):
        return self.main(x)
        

class DisentangleWeightNet(nn.Module):
    def __init__(self, nb_linear=4, input_dim=600):
        super(DisentangleWeightNet, self).__init__()

        self.hidden_dim = 512
        self.input_dim = input_dim

        if nb_linear == 2:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 4
                nn.Linear(self.hidden_dim, 1)
            )
        elif nb_linear == 3:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                #nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                # 2
                nn.Linear(self.hidden_dim, self.hidden_dim),
                #nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),
                # 4
                nn.Linear(self.hidden_dim, 1)
            )
        else:
            self.main = nn.Sequential(
                # 1
                nn.Linear(self.input_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 2
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 3
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(inplace=True),

                # 4
                nn.Linear(self.hidden_dim, 1)
            )

    def forward(self, x):
        return F.relu(self.main(x))



if __name__ == "__main__":

    img_size = 128
    x = torch.zeros((64, 3, img_size, img_size))
    enc_sh = BaseEncoder(
        img_size=img_size, in_channels=3, num_filters=16, kernel_size=1, repr_dim=64
    )
    enc_ex = BaseEncoder(
        img_size=img_size,
        in_channels=3,
        num_filters=16,
        kernel_size=1,
        repr_dim=64,
    )

    sh_repr, sh_feature = enc_sh(x)
    ex_repr, ex_feature = enc_ex(x)
    merge_repr = torch.cat([sh_repr, ex_repr], dim=1)
    merge_feature = torch.cat([sh_feature, ex_feature], dim=1)
    concat_repr = tile_and_concat(tensor=merge_feature, vector=merge_repr)
    t_loc = LocalStatisticsNetwork(img_feature_channels=concat_repr.size(1))
    t_glob = GlobalStatisticsNetwork(
        feature_map_size=merge_feature.size(2),
        feature_map_channels=merge_feature.size(1),
        latent_dim=merge_repr.size(1),
    )
    print(t_glob(feature_map=merge_feature, representation=merge_repr).shape)
    # print(b[0])
    # # print(b[0].shape)
