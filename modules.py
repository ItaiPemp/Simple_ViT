import torch
from einops.layers.torch import Rearrange
from torch import nn


class PatchEmbeddings(nn.Module):
    def __init__(self, d_model, patch_size, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, stride=patch_size,
                              kernel_size=patch_size)  # This is equivalent to splitting the image into patches and doing a linear transformation on each patch.

    def forward(self, x):  # x is image, shape (B,C,H,W)
        x = self.conv(x)  # shape is (B,d_model,H//patch_size,W//patch_size)
        x = Rearrange('b d h w -> b (h w) d')(x)  # shape is batch_size, num_patches, d_model
        return x


class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model, num_patches):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(1, num_patches+1, d_model),
                                                 requires_grad=True)  # +1 for cls token

    def forward(self, x):  # assume x is after patchEmbeddings(b,num_patches,d_model)
        return x + self.positional_encodings


class ClassificationHead(nn.Module):
    def __init__(self, d_model, n_hidden, n_classes, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_classes)
        self.activation = nn.GELU()  # could be replaced with ReLU,silu,etc.

    def forward(self, x):
        # we dont need softmax here because we use cross entropy loss which applies softmax for us
        x = self.dropout(self.activation(self.linear1(x)))
        return self.linear2(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout=0.2):
        super(TransformerEncoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding (d_model)
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),  # d_model -> d_model*4 ->d_model
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, max_sequence(num patches), d_model)

        """
        norm1_out = self.norm1(x)
        attention_out, _ = self.attention(norm1_out, norm1_out, norm1_out)  # batch*max_sequence*d_model
        attention_residual_out = self.dropout1(attention_out + x)  # same dimention. add residual connections
        norm2_out = self.norm2(attention_residual_out)  # batch*max_sequence*d_model

        feed_fwd_out = self.feed_forward(
            norm2_out)  # batch*max_sequence*d_model -> ##batch*max_sequence*(d_model*4) -> #batch*max_sequence*d_model
        feed_fwd_residual_out = feed_fwd_out + attention_residual_out  # batch*max_sequence*d_model
        out = self.dropout2(feed_fwd_residual_out)  # batch*max_sequence*d_model

        return out


class VisionTransformer(nn.Module):

    def __init__(self, patch_size,num_patches, num_classes, d_model, num_layers=8, expansion_factor=4, n_heads=8, dropout=0.0):
        super(VisionTransformer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.patch_emb = PatchEmbeddings(d_model, patch_size, 3)  # 3 is the number of channels in the image
        self.positional_emb = PositionalEmbeddings(d_model, num_patches)
        self.classification_head = ClassificationHead(d_model, d_model*8, num_classes, dropout)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, expansion_factor, n_heads,dropout) for i in range(num_layers)])

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, channels, height, width)

        """
        x = self.patch_emb(x)
        cls_token = self.cls_token_emb.expand(x.shape[0], -1, -1)  # batch_size, 1, d_model
        x = torch.cat([cls_token, x], dim=1)  # batch_size, num_patches+1(cls token is first), d_model
        x = self.positional_emb(x)  # batch_size, num_patches+1, d_model
        #x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        cls_token_final = x[:, 0, :]  # batch_size, d_model of cls token
        out = self.classification_head(cls_token_final)  # batch_size, num_classes
        return out
