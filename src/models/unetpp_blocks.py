import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)  # drop probability

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x  # at inference, or when the drop probability is 0, return the input unchanged
        keep = 1.0 - self.drop_prob  # keep probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # mask shape: same as the batch dimension, 1 everywhere else
        mask = x.new_empty(shape).bernoulli_(keep) / keep  # Bernoulli mask, rescaled to preserve the expectation
        return x * mask  # apply the mask, giving stochastic path dropping


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        squeezed = max(1, ch // r)  # squeezed channel count, kept from dropping below 1
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling, giving a channel descriptor
        self.fc = nn.Sequential(
            nn.Conv2d(ch, squeezed, 1),  # channel reduction
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, ch, 1),  # channel expansion
            nn.Sigmoid()  # normalized to [0,1] to serve as weights
        )

    def forward(self, x):
        w = self.fc(self.pool(x))  # compute the channel weights
        return x * w  # weight the channel features


class CBAMBlock(nn.Module):
    def __init__(self, ch, r=16, kernel=7):
        super().__init__()
        squeezed = max(1, ch // r)  # squeezed channel count
        self.avg = nn.AdaptiveAvgPool2d(1)  # channel average pooling
        self.max = nn.AdaptiveMaxPool2d(1)  # channel max pooling
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, squeezed, 1),  # channel reduction
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, ch, 1)  # channel expansion
        )
        self.sigmoid = nn.Sigmoid()  # activation function
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False),  # spatial attention convolution
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))  # channel attention weights
        x = x * ca  # channel weighting
        sa = self.spatial(torch.cat([
            torch.mean(x, 1, keepdim=True),  # channel mean as a spatial descriptor
            torch.max(x, 1, keepdim=True)[0]  # channel max as a spatial descriptor
        ], dim=1))
        return x * sa  # the output after spatial weighting


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)  # convolution layer, no bias
        self.bn = nn.BatchNorm2d(out_ch)  # batch normalization
        self.act = nn.ReLU(inplace=True)  # activation function

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # convolution -> normalization -> activation


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, cbam=False, dropp=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, 1, 1)  # 1st convolution block
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, 1, 1)  # 2nd convolution block
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()  # input projection (to match the channels)
        self.se = SEBlock(out_ch) if se else nn.Identity()  # SE attention module (optional)
        self.cbam = CBAMBlock(out_ch) if cbam else nn.Identity()  # CBAM attention module (optional)
        self.dp = DropPath(dropp)  # stochastic path dropping

    def forward(self, x):
        idn = self.proj(x)  # projected residual connection
        out = self.conv2(self.conv1(x))  # two convolution layers
        out = self.se(self.cbam(out))  # attention weighting
        out = idn + self.dp(out)  # residual sum plus stochastic dropping
        return F.relu(out, inplace=True)  # final ReLU activation


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, atrous=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1 if d == 1 else 3,
                          padding=0 if d == 1 else d, dilation=d, bias=False),  # dilated convolution
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for d in atrous
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(atrous), out_ch, 1, bias=False),  # fuse all the branches
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [br(x) for br in self.branches]  # multi-scale dilated features
        x = torch.cat(feats, dim=1)  # channel concatenation
        return self.project(x)  # fused output


class Up(nn.Module):
    def __init__(self, dec_ch, skip_ch, out_ch, bilinear=True, se=False, cbam=False, dropp=0.0):
        super().__init__()
        self.bilinear = bool(bilinear)
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # bilinear upsampling
        else:
            self.up = nn.ConvTranspose2d(dec_ch, dec_ch, 2, 2)  # transposed-convolution upsampling

        in_ch = dec_ch + skip_ch  # channel count after concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),  # 1x1 convolution fusing the channels
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.block = ResBlock(out_ch, out_ch, se=se, cbam=cbam, dropp=dropp)  # residual convolution block

    def forward(self, x_dec, x_skip):
        x_up = self.up(x_dec)  # upsample the decoder features
        diffY = x_skip.size(2) - x_up.size(2)  # height difference
        diffX = x_skip.size(3) - x_up.size(3)  # width difference
        if diffY != 0 or diffX != 0:
            x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])  # pad so that the sizes match
        x = torch.cat([x_skip, x_up], dim=1)  # concatenate the skip connection and the upsampled features
        x = self.fuse(x)  # fuse the channels
        return self.block(x)  # processed by the residual convolution block


class UNetPP(nn.Module):
    def __init__(self, in_ch=64, base=64, num_classes=1, bilinear=True,
                 use_se=True, use_cbam=False, use_aspp=True, droppath_prob=0.0):
        super().__init__()
        b = base
        se = use_se
        cb = use_cbam
        dp = float(droppath_prob)
        self.bilinear = bool(bilinear)

        self.pool = nn.MaxPool2d(2)  # max-pooling downsampling

        # 5 residual blocks in the encoder, doubling the channels layer by layer
        self.conv00 = ResBlock(in_ch, b, se=se, cbam=cb, dropp=dp)
        self.conv10 = ResBlock(b, b * 2, se=se, cbam=cb, dropp=dp)
        self.conv20 = ResBlock(b * 2, b * 4, se=se, cbam=cb, dropp=dp)
        self.conv30 = ResBlock(b * 4, b * 8, se=se, cbam=cb, dropp=dp)
        self.conv40 = ResBlock(b * 8, b * 16, se=se, cbam=cb, dropp=dp)

        self.aspp = ASPP(b * 16, b * 16) if use_aspp else nn.Identity()  # atrous spatial pyramid pooling

        # multi-stage upsampling modules in the decoder, fusing skip connections from several levels
        self.up01 = Up(b * 2, b, b, bilinear, se, cb, dp)
        self.up11 = Up(b * 4, b * 2, b * 2, bilinear, se, cb, dp)
        self.up21 = Up(b * 8, b * 4, b * 4, bilinear, se, cb, dp)
        self.up31 = Up(b * 16, b * 8, b * 8, bilinear, se, cb, dp)

        self.up02 = Up(b * 2, b + b, b, bilinear, se, cb, dp)
        self.up12 = Up(b * 4, b * 2 + b * 2, b * 2, bilinear, se, cb, dp)
        self.up22 = Up(b * 8, b * 4 + b * 4, b * 4, bilinear, se, cb, dp)

        self.up03 = Up(b * 2, b + b + b, b, bilinear, se, cb, dp)
        self.up13 = Up(b * 4, b * 2 + b * 2 + b * 2, b * 2, bilinear, se, cb, dp)

        self.up04 = Up(b * 2, b * 4, b, bilinear, se, cb, dp)

        self.outc = nn.Conv2d(b, num_classes, 1)  # 1x1 convolution producing the prediction

    def forward(self, x):
        x00 = self.conv00(x)  # encoder level 0
        x10 = self.conv10(self.pool(x00))  # encoder level 1
        x20 = self.conv20(self.pool(x10))  # encoder level 2
        x30 = self.conv30(self.pool(x20))  # encoder level 3
        x40 = self.conv40(self.pool(x30))  # encoder level 4
        x40 = self.aspp(x40)  # ASPP enhancement

        # decoder stage, the dense skip-connection structure of UNet++
        x31 = self.up31(x40, x30)
        x21 = self.up21(x30, x20)
        x11 = self.up11(x20, x10)
        x01 = self.up01(x10, x00)

        x22 = self.up22(x31, torch.cat([x20, x21], dim=1))
        x12 = self.up12(x21, torch.cat([x10, x11], dim=1))
        x02 = self.up02(x11, torch.cat([x00, x01], dim=1))

        x13 = self.up13(x22, torch.cat([x10, x11, x12], dim=1))
        x03 = self.up03(x12, torch.cat([x00, x01, x02], dim=1))

        x04 = self.up04(x13, torch.cat([x00, x01, x02, x03], dim=1))

        logits = self.outc(x04)  # final output
        return logits


class ECABlock(nn.Module):
    def __init__(self, ch, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # channel average pooling
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)  # 1D convolution
        self.sigmoid = nn.Sigmoid()  # activation function

    def forward(self, x):
        y = self.pool(x)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(1, 2))  # convolution along the channel dimension
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # activate and restore the shape
        return x * y  # channel weighting


class SCSEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        squeeze = max(1, ch // r)  # channel squeezing
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # channel attention
            nn.Conv2d(ch, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, ch, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(ch, 1, 1),  # spatial attention
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)  # channel attention and spatial attention added together


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, ch, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch)  # depthwise convolution
        self.norm = nn.LayerNorm(ch, eps=1e-6)  # LayerNorm, applied over the channel dimension
        self.pw1 = nn.Linear(ch, 4 * ch)  # channel expansion
        self.act = nn.GELU()  # activation
        self.pw2 = nn.Linear(4 * ch, ch)  # channel compression
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(ch)) if layer_scale_init_value > 0 else None  # layer-scale parameter
        self.dp = DropPath(drop_path)  # stochastic path dropping

    def forward(self, x):
        shortcut = x  # residual connection
        x = self.dw(x)  # depthwise convolution
        x = x.permute(0, 2, 3, 1)  # NHWC layout, as required by LayerNorm
        x = self.norm(x)  # normalization
        x = self.pw2(self.act(self.pw1(x)))  # MLP transform
        if self.gamma is not None:
            x = self.gamma * x  # scaling
        x = x.permute(0, 3, 1, 2)  # back to the NCHW layout
        x = shortcut + self.dp(x)  # residual connection plus stochastic dropping
        return x


class ResidualWrapper(nn.Module):
    def __init__(self, module, drop_prob=0.0):
        super().__init__()
        self.module = module  # the wrapped module
        self.dp = DropPath(drop_prob)  # stochastic path dropping

    def forward(self, x):
        return x + self.dp(self.module(x))  # residual connection plus stochastic dropping


class ResidualECA(nn.Module):
    def __init__(self, ch, k_size=3, drop_prob=0.0):
        super().__init__()
        self.block = ResidualWrapper(ECABlock(ch, k_size), drop_prob)  # ECA module with a residual connection

    def forward(self, x):
        return self.block(x)


class ResidualSCSE(nn.Module):
    def __init__(self, ch, r=16, drop_prob=0.0):
        super().__init__()
        self.block = ResidualWrapper(SCSEBlock(ch, r), drop_prob)  # SCSE module with a residual connection

    def forward(self, x):
        return self.block(x)