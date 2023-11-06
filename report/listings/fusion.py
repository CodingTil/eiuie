import torch
import numpy as np
from util import BGR2HSI, HSI2BGR


class ChannelNet(torch.nn.Module):
    def __init__(self):
        super(ChannelNet, self).__init__()
        self.fc = torch.nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc(x)


class FusionNet(torch.nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.h_net = ChannelNet()
        self.s_net = ChannelNet()
        self.i_net = ChannelNet()

    def forward(self, x):
        # Flatten the middle dimensions
        x = x.view(-1, 12)
        # Splitting the input for the three channels
        h_channel = x[:, 0::3]
        s_channel = x[:, 1::3]
        i_channel = x[:, 2::3]
        # Getting the outputs
        h_out = self.h_net(h_channel)
        s_out = self.s_net(s_channel)
        i_out = self.i_net(i_channel)
        # Concatenate the outputs to get the final output
        return torch.cat((h_out, s_out, i_out), dim=1)


def process_image(image: np.ndarray, net: FusionNet) -> np.ndarray:
    dimensions = image.shape
    um_imagge = unsharp_masking(image)
    um_image = BGR2HSI(um_imagge)
    hf_image = homomorphic_filtering(image)
    hf_image = BGR2HSI(hf_image)
    rtx_image = retinex(image)
    rtx_image = BGR2HSI(rtx_image)

    # Use numpy functions for efficient concatenation
    um_image = um_image.reshape(-1, 3)
    hf_image = hf_image.reshape(-1, 3)
    rtx_image = rtx_image.reshape(-1, 3)
    all_inputs = np.hstack([um_image, hf_image, rtx_image])
    all_inputs = torch.tensor(all_inputs, dtype=torch.float32)

    # Model inference
    outputs = net(all_inputs).numpy()
    outputs = np.clip(outputs, 0, 1)
    fused_image = outputs.reshape(dimensions[0], dimensions[1], 3)
    fused_image = HSI2BGR(fused_image)
    return fused_image
