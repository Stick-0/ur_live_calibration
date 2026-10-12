# robotpose/network.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def _attention_distinct_qkv(attention, query, key=None):
    """Reuse normalized values without changing MultiheadAttention's Q/K/V path."""
    if key is None:
        key = query
    # The original code produced three distinct tensors by calling LayerNorm
    # three times. Keep that identity relationship: passing the same object
    # would select different projection/attention kernels in PyTorch.
    return attention(query, key.view_as(key), key.view_as(key))[0]


class PatchEmbedding(nn.Module):
    def __init__(self, w_size=224, h_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (w_size // patch_size) * (h_size // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x


class FiLMConditioning(nn.Module):
    def __init__(self, transformer_embed_dim=768, cnn_channels=64):
        super().__init__()
        self.fc_gamma = nn.Linear(transformer_embed_dim, cnn_channels)
        self.fc_beta = nn.Linear(transformer_embed_dim, cnn_channels)

    def forward(self, transformer_out, cnn_feature):
        batch = transformer_out.size(0)
        x = transformer_out.mean(dim=1)  # (batch, embed_dim=768)

        gamma = self.fc_gamma(x).unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        beta = self.fc_beta(x).unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)

        out = gamma * cnn_feature + beta
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.training:
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        else:
            normalized = self.norm1(x)
            x = x + _attention_distinct_qkv(self.attn, normalized)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder_Split(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.norm1_1 = nn.LayerNorm(embed_dim)
        self.norm1_2 = nn.LayerNorm(embed_dim)
        self.norm1_3 = nn.LayerNorm(embed_dim)
        self.norm1_4 = nn.LayerNorm(embed_dim)
        self.norm1_5 = nn.LayerNorm(embed_dim)
        self.norm1_6 = nn.LayerNorm(embed_dim)
        self.norm1_7 = nn.LayerNorm(embed_dim)

        self.attn1 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.attn4 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.attn5 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.attn6 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)
        self.attn7 = nn.MultiheadAttention(embed_dim, 1, dropout=dropout, batch_first=True)

        self.norm2_1 = nn.LayerNorm(embed_dim)
        self.norm2_2 = nn.LayerNorm(embed_dim)
        self.norm2_3 = nn.LayerNorm(embed_dim)
        self.norm2_4 = nn.LayerNorm(embed_dim)
        self.norm2_5 = nn.LayerNorm(embed_dim)
        self.norm2_6 = nn.LayerNorm(embed_dim)
        self.norm2_7 = nn.LayerNorm(embed_dim)

        self.mlp1 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.mlp2 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.mlp3 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.mlp4 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.mlp5 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.mlp6 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.mlp7 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))

        self.norm3_1 = nn.LayerNorm(embed_dim)
        self.norm3_2 = nn.LayerNorm(embed_dim)
        self.norm3_3 = nn.LayerNorm(embed_dim)
        self.norm3_4 = nn.LayerNorm(embed_dim)
        self.norm3_5 = nn.LayerNorm(embed_dim)
        self.norm3_6 = nn.LayerNorm(embed_dim)
        self.norm3_7 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if self.training:
            x1 = x + self.attn1(self.norm1_1(x), self.norm1_1(x), self.norm1_1(x))[0]
            x2 = x + self.attn2(self.norm1_2(x), self.norm1_2(x), self.norm1_2(x))[0] \
                + self.attn2(self.norm1_2(x1), self.norm1_2(x), self.norm1_2(x))[0]
            x3 = x + self.attn3(self.norm1_3(x), self.norm1_3(x), self.norm1_3(x))[0] \
                + self.attn3(self.norm1_3(x2), self.norm1_3(x), self.norm1_3(x))[0]
            x4 = x + self.attn4(self.norm1_4(x), self.norm1_4(x), self.norm1_4(x))[0] \
                + self.attn4(self.norm1_4(x3), self.norm1_4(x), self.norm1_4(x))[0]
            x5 = x + self.attn5(self.norm1_5(x), self.norm1_5(x), self.norm1_5(x))[0] \
                + self.attn5(self.norm1_5(x4), self.norm1_5(x), self.norm1_5(x))[0]
            x6 = x + self.attn6(self.norm1_6(x), self.norm1_6(x), self.norm1_6(x))[0] \
                + self.attn6(self.norm1_6(x5), self.norm1_6(x), self.norm1_6(x))[0]
            x7 = x + self.attn7(self.norm1_7(x), self.norm1_7(x), self.norm1_7(x))[0] \
                + self.attn7(self.norm1_7(x6), self.norm1_7(x), self.norm1_7(x))[0]

            x6 = x6 + self.attn6(self.norm1_6(x7), self.norm1_6(x7), self.norm1_6(x7))[0]
            x5 = x5 + self.attn5(self.norm1_5(x6), self.norm1_5(x6), self.norm1_5(x6))[0]
            x4 = x4 + self.attn4(self.norm1_4(x5), self.norm1_4(x5), self.norm1_4(x5))[0]
            x3 = x3 + self.attn3(self.norm1_3(x4), self.norm1_3(x4), self.norm1_3(x4))[0]
            x2 = x2 + self.attn2(self.norm1_2(x3), self.norm1_2(x3), self.norm1_2(x3))[0]
            x1 = x1 + self.attn1(self.norm1_1(x2), self.norm1_1(x2), self.norm1_1(x2))[0]
        else:
            # Each branch reuses its normalization of the common input.
            # The order of residual additions and attention operations is unchanged.
            branches = []
            for index in range(1, 8):
                norm = getattr(self, f"norm1_{index}")
                attention = getattr(self, f"attn{index}")
                normalized = norm(x)
                branch = x + _attention_distinct_qkv(attention, normalized)
                if branches:
                    branch = branch + _attention_distinct_qkv(
                        attention, norm(branches[-1]), normalized
                    )
                branches.append(branch)

            for index in range(6, 0, -1):
                norm = getattr(self, f"norm1_{index}")
                attention = getattr(self, f"attn{index}")
                branches[index - 1] = branches[index - 1] + _attention_distinct_qkv(
                    attention, norm(branches[index])
                )
            x1, x2, x3, x4, x5, x6, x7 = branches

        x1 = x1 + self.mlp1(self.norm2_1(x1))
        x2 = x2 + self.mlp2(self.norm2_2(x2))
        x3 = x3 + self.mlp3(self.norm2_3(x3))
        x4 = x4 + self.mlp4(self.norm2_4(x4))
        x5 = x5 + self.mlp5(self.norm2_5(x5))
        x6 = x6 + self.mlp6(self.norm2_6(x6))
        x7 = x7 + self.mlp7(self.norm2_7(x7))

        x1 = self.norm3_1(x1)
        x2 = self.norm3_2(x2)
        x3 = self.norm3_3(x3)
        x4 = self.norm3_4(x4)
        x5 = self.norm3_5(x5)
        x6 = self.norm3_6(x6)
        x7 = self.norm3_7(x7)

        return x1, x2, x3, x4, x5, x6, x7


class VisionTransformer(nn.Module):
    def __init__(self, w_size=224, h_size=224, patch_size=8, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(w_size, h_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads) for _ in range(depth)])
        self.transformer_split = TransformerEncoder_Split(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x1, x2, x3, x4, x5, x6, x7 = self.transformer_split(x)
        return x1, x2, x3, x4, x5, x6, x7


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, k, 1, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.merge_conv = ConvBNReLU(c_in + (c_skip or 0), c_out)
    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.merge_conv(x)
        return x



class KPHead(nn.Module):
    def __init__(self, film: FiLMConditioning, base_channels=64):
        super().__init__()
        self.film = film
        self.conv = ConvBNReLU(base_channels, base_channels)
        self.up = UpBlock(c_in=base_channels, c_skip=0, c_out=base_channels)  # 240x320 -> 480x640
        self.head_seg = nn.Conv2d(base_channels, 1, 1)
        self.head_vec = nn.Conv2d(base_channels, 2, 1)

    def forward(self, shared_240, xk_tokens):   
        y = self.film(xk_tokens, shared_240)              # (B,C,240,320)
        y = self.conv(y)
        y = self.up(y)                        # (B,C,480,640)
        seg_k = self.head_seg(y)              # (B,1,480,640)
        vec_k = self.head_vec(y)              # (B,2,480,640)
        return seg_k, vec_k


class RobotJointDetection(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights="DEFAULT")
        return_nodes = {
            'relu':   'feat1',   # (B, 64, 240,320)
            'layer1': 'feat2',
            'layer2': 'feat4',
            'layer3': 'feat8',
            # 'layer4': 'feat16'
        }

        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)
    
        self.vit = VisionTransformer(w_size=40, h_size=30, in_channels=256, depth=12)  # in=256

        # (2) 공유 U-Net 업패스: 30x40→60x80→120x160→240x320
        self.bottleneck = ConvBNReLU(256, 256)                 # 256 -> 256
        self.up1 = UpBlock(c_in=256, c_skip=128, c_out=128)  # 256 + 128 -> 128 (60x80)
        self.up2 = UpBlock(c_in=128, c_skip=64, c_out=64)  # 128 + 64  -> 64  (120x160)
        self.up3 = UpBlock(c_in=64, c_skip=64,      c_out=64) # 64 + 64    -> 64  (240x320)
        
        # (3) 분기 헤드: FiLM/Head 채널을 분기 입력(64)에 맞춤
        self.kp_films = nn.ModuleList([FiLMConditioning(768, 64) for _ in range(7)])
        self.kp_heads = nn.ModuleList([KPHead(self.kp_films[k], 64) for k in range(7)])
   

    def forward(self, x_img: torch.Tensor):
        x = self.encoder(x_img)
        feat1 = x['feat1']; feat2 = x['feat2']; feat4 = x['feat4']; feat8 = x['feat8']

        # ViT
        x1, x2, x3, x4, x5, x6, x7 = self.vit(feat8)

        # 공유 업패스
        y = self.bottleneck(feat8)
        y = self.up1(y, skip=feat4)
        y = self.up2(y, skip=feat2)
        shared_240 = self.up3(y, skip=feat1)

        # 7개 분기 (히트맵 완전 미사용)
        tokens = [x1, x2, x3, x4, x5, x6, x7]
        seg_list, vec_list = [], []
        for k in range(7):
            seg_k, vec_k = self.kp_heads[k](shared_240, tokens[k])
            seg_list.append(seg_k)
            vec_list.append(vec_k)

        seg_logits = torch.cat(seg_list, dim=1)   # (B,7,480,640)
        vec_field  = torch.stack(vec_list, dim=1) # (B,7,2,480,640)
        return seg_logits, vec_field


