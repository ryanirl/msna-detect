import torch
import torch.nn as nn


def ConvBlock(in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same"):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace = True)
    )


def MiddleBlock(in_channels, out_channels):
    return nn.Sequential(
        ConvBlock(in_channels, out_channels),
        nn.Dropout(p = 0.2),
        ConvBlock(out_channels, out_channels)
    )


def FinalBlock(in_channels, out_channels):
    return nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = "same")


class ResidualBlock(nn.Module):
    """ Residual encoder block. """
    def __init__(self, in_channels, feature_maps, stride = 1, downsample = None, pooling_size = 2):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = pooling_size, stride = None, ceil_mode = True)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, feature_maps, kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm1d(feature_maps)
        )

        self.conv1 = nn.Conv1d(in_channels, feature_maps, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm1d(feature_maps)

        self.conv2 = nn.Conv1d(feature_maps, feature_maps, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm1d(feature_maps)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity
        skip_connection = self.relu(x)
        x = self.maxpool(skip_connection)

        return x, skip_connection


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p = 2):
        super(DecoderBlock, self).__init__()

        self.conv_block_0 = ConvBlock(in_channels + out_channels, out_channels)
        self.conv_block_1 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        # Dynamic upsampling based on skip connection dimensions
        x = nn.functional.interpolate(
            x, 
            size = skip_connection.shape[-1],
            mode = "linear",
            #mode = "nearest",
            align_corners = False
        )

        x = torch.cat((x, skip_connection), 1)
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(Encoder, self).__init__()

        self.encoder_0 = ResidualBlock(in_channels, 32)
        self.encoder_1 = ResidualBlock(32, 64)
        self.encoder_2 = ResidualBlock(64, 128)
        self.encoder_3 = ResidualBlock(128, 128)
        self.encoder_4 = ResidualBlock(128, 128)
        self.encoder_5 = ResidualBlock(128, 128)
        self.encoder_6 = ResidualBlock(128, 128)

        self.middle = MiddleBlock(in_channels = 128, out_channels = 128)

    def forward(self, x):
        x, x0 = self.encoder_0(x)
        x, x1 = self.encoder_1(x)
        x, x2 = self.encoder_2(x)
        x, x3 = self.encoder_3(x)
        x, x4 = self.encoder_4(x)
        x, x5 = self.encoder_5(x)
        x, x6 = self.encoder_6(x)

        x7 = self.middle(x)

        return [x0, x1, x2, x3, x4, x5, x6, x7]

    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_0 = DecoderBlock(128, 128) 
        self.decoder_1 = DecoderBlock(128, 128) 
        self.decoder_2 = DecoderBlock(128, 128) 
        self.decoder_3 = DecoderBlock(128, 128) 
        self.decoder_4 = DecoderBlock(128, 128) 
        self.decoder_5 = DecoderBlock(128, 64)
        self.decoder_6 = DecoderBlock(64, 32)

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7):
        x = self.decoder_0(x7, x6)
        x = self.decoder_1(x,  x5)
        x = self.decoder_2(x,  x4)
        x = self.decoder_3(x,  x3)
        x = self.decoder_4(x,  x2)
        x = self.decoder_5(x,  x1)
        x = self.decoder_6(x,  x0)

        return x 
    

class Unet1d(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()

        self.backbone = Encoder(in_channels)
        self.head = Decoder()
        self.final = FinalBlock(32, 1)

        self.downsample_factor = 2 ** 7
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal.
        """
        out = self.backbone(x)

        x = self.head(*out)
        x = self.final(x)

        # Forced temperature of 0.1
        x = torch.sigmoid(x / 0.1)

        return x


