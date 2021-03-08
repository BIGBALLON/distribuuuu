"""
Bottleneck Transformers for Visual Recognition.
adapted from https://github.com/CandiceD17/Bottleneck-Transformers-for-Visual-Recognition
"""
import torch
from einops import rearrange
from torch import einsum, nn

try:
    from distribuuuu.models import resnet50
except ImportError:
    from torchvision.models import resnet50


def expand_dim(t, dim, k):
    """
    Expand dims for t at dim to k
    """
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    """
    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    Tensowflow source code in the appendix of: https://arxiv.org/pdf/1904.09925.pdf
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = torch.zeros((B, Nh, L, 1)).cuda()
    x = torch.cat((x, col_pad), dim=3)
    flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
    flat_pad = torch.zeros((B, Nh, L - 1)).cuda()
    flat_x = torch.cat((flat_x, flat_pad), dim=2)
    # Reshape and slice out the padded elements
    final_x = torch.reshape(flat_x, (B, Nh, L + 1, 2 * L - 1))
    return final_x[:, :, :L, L - 1 :]


def relative_logits_1d(q, rel_k):
    """
    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.
    The details of relative position is explained in: https://arxiv.org/pdf/1803.02155.pdf
    """
    B, Nh, H, W, _ = q.shape
    rel_logits = torch.einsum("b n h w d, m d -> b n h w m", q, rel_k)
    # Collapse height and heads
    rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class AbsPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        assert height == width
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, "h d -> h () d") + rearrange(
            self.width, "w d -> () w d"
        )
        emb = rearrange(emb, " h w d -> (h w) d")
        logits = einsum("b h i d, j d -> b h i j", q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        assert height == width
        scale = dim_head ** -0.5
        self.fmap_size = height
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h = w = self.fmap_size

        q = rearrange(q, "b h (x y) d -> b h x y d", x=h, y=w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b h x i y j-> b h (x y) (i j)")

        q = rearrange(q, "b h x y d -> b h y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b h x i y j -> b h (y x) (j i)")
        return rel_logits_w + rel_logits_h


class BoTBlock(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out,
        stride=1,
        heads=4,
        proj_factor=4,
        dim_qk=128,
        dim_v=128,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()
        if dim != dim_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim_out),
                activation,
            )
        else:
            self.shortcut = nn.Identity()

        bottleneck_dimension = dim_out // proj_factor  # from 2048 to 512
        attn_dim_out = heads * dim_v

        self.net = nn.Sequential(
            nn.Conv2d(dim, bottleneck_dimension, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_dimension),
            activation,
            MHSA(
                dim=bottleneck_dimension,
                fmap_size=fmap_size,
                heads=heads,
                dim_qk=dim_qk,
                dim_v=dim_v,
                rel_pos_emb=rel_pos_emb,
            ),
            nn.AvgPool2d((2, 2)) if stride == 2 else nn.Identity(),  # same padding
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(dim_out),
        )

        nn.init.zeros_(
            self.net[-1].weight
        )  # last batch norm uses zero gamma initializer
        self.activation = activation

    def forward(self, featuremap):
        shortcut = self.shortcut(featuremap)
        featuremap = self.net(featuremap)
        featuremap += shortcut
        return self.activation(featuremap)


class MHSA(nn.Module):
    def __init__(
        self, dim, fmap_size, heads=4, dim_qk=128, dim_v=128, rel_pos_emb=False
    ):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2d(
            dim, out_channels_qk * 2, 1, bias=False
        )  # 1*1 conv to compute q, k
        self.to_v = nn.Conv2d(
            dim, out_channels_v, 1, bias=False
        )  # 1*1 conv to compute v
        self.softmax = nn.Softmax(dim=-1)

        height, width = fmap_size
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        heads = self.heads
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, dim=1)
        v = self.to_v(featuremap)
        q, k, v = map(
            lambda x: rearrange(x, "B (h d) H W -> B h (H W) d", h=heads), (q, k, v)
        )

        q *= self.scale

        logits = einsum("b h x d, b h y d -> b h x y", q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum("b h x y, b h y d -> b h x d", weights, v)
        attn_out = rearrange(attn_out, "B h (H W) d -> B (h d) H W", H=H)

        return attn_out


class BoTStack(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size,
        dim_out=2048,
        heads=4,
        proj_factor=4,
        num_layers=3,
        stride=2,
        dim_qk=128,
        dim_v=128,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out

            fmap_divisor = 2 if stride == 2 and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                BoTBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    stride=stride if is_first else 1,
                    heads=heads,
                    proj_factor=proj_factor,
                    dim_qk=dim_qk,
                    dim_v=dim_v,
                    rel_pos_emb=rel_pos_emb,
                    activation=activation,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f"assert {c} == self.dim {self.dim}"
        assert h == self.fmap_size[0] and w == self.fmap_size[1]
        return self.net(x)


def botnet50(pretrained=False, **kwargs):
    """
    Bottleneck Transformers for Visual Recognition.
    https://arxiv.org/abs/2101.11605
    """
    resnet = resnet50(pretrained, **kwargs)
    layer = BoTStack(dim=1024, fmap_size=(14, 14), stride=1, rel_pos_emb=True)
    backbone = list(resnet.children())
    model = nn.Sequential(
        *backbone[:-3],
        layer,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(2048, 1000),
    )
    return model


def test():
    x = torch.ones(16, 3, 224, 224)
    model = botnet50()
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
