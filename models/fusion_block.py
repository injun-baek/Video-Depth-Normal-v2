import torch.nn as nn


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FusionLayer(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        lhs_channels,
        out_channels,
        activation,
        bn=False,
        align_corners=True,
        rhs_size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.align_corners = align_corners

        self.in_channels = lhs_channels
        self.out_channels = out_channels

        self.lhs_in_conv = nn.Conv2d(
            lhs_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.lhs_res_block = ResidualConvUnit(out_channels, activation, bn)
        self.rhs_res_block = ResidualConvUnit(out_channels, activation, bn)
        self.fusion_res_block = ResidualConvUnit(out_channels, activation, bn)

        self.out_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

        if rhs_size is None:
            self.lhs_modifier = {"scale_factor": 2}
        else:
            self.lhs_modifier = {"size": rhs_size}


    def forward(self, lhs_x, rhs_x=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        
        B_l, S_l, C_l, H_l, W_l = lhs_x.shape
        lhs_x_reshaped = lhs_x.view(B_l * S_l, C_l, H_l, W_l)
        
        if rhs_x is not None:
            B_r, S_r, C_r, H_r, W_r = rhs_x.shape
            rhs_x_reshaped = rhs_x.view(B_r * S_r, C_r, H_r, W_r)
        
        lhs_x_reshaped = nn.functional.interpolate(
            lhs_x_reshaped.contiguous(), **self.lhs_modifier, mode="bilinear", align_corners=self.align_corners
        )
        
        _, _, H_out, W_out= lhs_x_reshaped.shape
        lhs_x_reshaped = self.lhs_in_conv(lhs_x_reshaped)
        output = self.lhs_res_block(lhs_x_reshaped)
        if rhs_x is not None:
            output = output + self.rhs_res_block(rhs_x_reshaped)  # todo: test using cat
        output = self.fusion_res_block(output)
        output = self.out_conv(output)

        return output.view(B_l, S_l, self.out_channels, H_out, W_out)
