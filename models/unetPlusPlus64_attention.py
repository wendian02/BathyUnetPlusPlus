import torch.nn as nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_model = smp.UnetPlusPlus(
            encoder_name="resnet34",  
            encoder_weights=None,  
            in_channels=8,  # Planet 8 bands
            classes=1,  # Depth output
            activation=None,
            decoder_attention_type="scse"  # Use spatial and channel attention mechanism
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
        self.center = slice(16, 48)

    def forward(self, outputs, targets):
        _, _, H, W = outputs.shape
        if H < self.center.stop or W < self.center.stop:
            raise ValueError(
                f"Input size ({H}x{W}) is too small: must both be >= {self.center.stop}")

        # (B, 1, 64, 64)
        o_center = outputs[:, :, self.center, self.center]
        t_center = targets[:, :, self.center, self.center]

        valid_mask = torch.isfinite(t_center)
        # only remain valid
        valid_outputs = o_center[valid_mask]
        valid_targets = t_center[valid_mask]
        
        # var in mse_loss mustn't be nan
        loss = F.mse_loss(valid_outputs, valid_targets)

        return loss


if __name__ == '__main__':
    inputs = torch.ones((4,8,64,64))
    depth_model = get_model()
    outputs = depth_model(inputs)
    print(outputs.shape)


