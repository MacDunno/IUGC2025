import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath, Mlp
from torchvision.models import resnet50


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTBottleneck(nn.Module):
    def __init__(self, in_channels, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, (16*16)+1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj_in(x).flatten(2).transpose(1,2)  # B, N=H*W, embed_dim
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:, :].transpose(1, 2).reshape(B, -1, H, W)
        return x

class TransUNet(nn.Module):
    def __init__(self, num_keypoints=3, heatmap_size=64, embed_dim=192):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.encoder1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # 256 ch
        self.encoder2 = backbone.layer2  # 512 ch
        self.encoder3 = backbone.layer3  # 1024 ch
        self.encoder4 = backbone.layer4  # 2048 ch
        self.vit_bottleneck = ViTBottleneck(in_channels=2048, embed_dim=embed_dim)
        self.up4 = nn.ConvTranspose2d(embed_dim, 1024, kernel_size=2, stride=2)
        self.dec4 = double_conv(2048, 1024)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = double_conv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        e0 = self.encoder0(x)  # B,64,H/2,W/2
        e1 = self.encoder1(e0) # B,256,H/4,W/4
        e2 = self.encoder2(e1) # B,512,H/8,W/8
        e3 = self.encoder3(e2) # B,1024,H/16,W/16
        e4 = self.encoder4(e3) # B,2048,H/32,W/32
        b = self.vit_bottleneck(e4)  # B, embed_dim, H/32, W/32
        d4 = self.up4(b)  # B,1024,H/16,W/16
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)  # B,512,H/8,W/8
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)  # B,256,H/4,W/4
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)  # B,64,H/2,W/2
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)
        heatmaps = self.final_layer(d1)
        if heatmaps.size(2) != self.heatmap_size or heatmaps.size(3) != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=(self.heatmap_size, self.heatmap_size),
                                    mode='bilinear', align_corners=True)
        return heatmaps

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


def get_deit_heatmap_model(num_keypoints=3, heatmap_size=64):
    return TransUNet(num_keypoints=num_keypoints, heatmap_size=heatmap_size)


def extract_coordinates(heatmaps, original_img_size=512):
    batch_size, num_keypoints, height, width = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = torch.div(max_indices, width, rounding_mode='floor').float() / height
    x_coords = (max_indices % width).float() / width
    coords = torch.zeros(batch_size, num_keypoints * 2, device=heatmaps.device)
    for i in range(num_keypoints):
        coords[:, i*2] = x_coords[:, i]
        coords[:, i*2+1] = y_coords[:, i]
    return coords


class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred_heatmaps, target_heatmaps):
        return self.criterion(pred_heatmaps, target_heatmaps)

def euclidean_distance(pred, target):
    pred = pred.view(-1, 3, 2) 
    target = target.view(-1, 3, 2)
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=2))  