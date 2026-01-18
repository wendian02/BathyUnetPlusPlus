import torch.nn as nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_model = smp.Unet(
            encoder_name="resnet34",  
            encoder_weights=None,  
            in_channels=8,  # Planet 8 bands
            classes=1,  # Depth output
            activation=None,
        )

    def forward(self, x):
        return self.depth_model(x)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # targets: (B, 1, H, W)
        mask = torch.isfinite(targets)
        if mask.sum() == 0:
            raise ValueError("No valid pixels in the batch")
        valid_outputs = outputs[mask]
        valid_targets = targets[mask]
        loss = F.mse_loss(valid_outputs, valid_targets)
        return loss

class MaskedCenterMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = slice(8, 24)

    def forward(self, outputs, targets):
        _, _, H, W = outputs.shape
        if H < self.center.stop or W < self.center.stop:
            raise ValueError(
                f"Input size ({H}x{W}) is too small: must both be >= {self.center.stop}")

        # (B, 1, 32, 32)
        o_center = outputs[:, :, self.center, self.center]
        t_center = targets[:, :, self.center, self.center]

        valid_mask = torch.isfinite(t_center)
        # only remain valid
        valid_outputs = o_center[valid_mask]
        valid_targets = t_center[valid_mask]

        # var in mse_loss mustn't be nan!
        loss = F.mse_loss(valid_outputs, valid_targets)

        return loss


if __name__ == '__main__':
    inputs = torch.ones((4,8,32,32))
    depth_model = get_model()
    outputs = depth_model(inputs)
    # print(depth_model)

    outputs = torch.randn((4,1,32,32))
    targets = torch.randn((4,1,32,32))
    targets[1,:,:,:] = torch.nan

    loss_fn = MaskedCenterMSELoss()
    loss = loss_fn(outputs, targets)
    print(loss)

    # load
    # depth_model = get_model()
    # depth_model.load_state_dict(torch.load("logs/logs_l8depth_unet_dongsha_cloud70/epoch_1.pth"))

