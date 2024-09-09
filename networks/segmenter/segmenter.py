import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.segmenter.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from networks.segmenter.vit import VisionTransformer
from networks.segmenter.decoder import MaskTransformer


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


class Segmenter_Ly(nn.Module):
    def __init__(
            self,
            img_size=384,
            patch_size=8,
            in_chans=3,
            out_chans=3,
            embed_dim=96,
            decoder_dim=96,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            p=0.,
            attn_p=0.):
        super(Segmenter_Ly, self).__init__()
        self.encoder = VisionTransformer(image_size=[img_size, img_size],
                                         patch_size=patch_size,
                                         channels=in_chans,
                                         n_layers=depth,
                                         d_model=embed_dim,
                                         d_ff=embed_dim*mlp_ratio,
                                         n_heads=n_heads,
                                         n_cls=1)
        self.decoder = MaskTransformer(n_cls=out_chans,
                                       patch_size=patch_size,
                                       d_encoder=embed_dim,
                                       n_layers=depth,
                                       n_heads=n_heads,
                                       d_model=decoder_dim,
                                       d_ff=mlp_ratio*mlp_ratio,
                                       drop_path_rate=p,
                                       dropout=attn_p)

    def forward(self, x):
        ori_W, ori_H = x.shape[2], x.shape[3]
        x = self.encoder(x, return_features=True)
        x = self.decoder(x[:, 1:], [ori_W, ori_H])
        x = F.interpolate(x, size=(ori_W, ori_H), mode="bilinear")
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    img_size = 640
    model = Segmenter_Ly(img_size=img_size, patch_size=8, depth=4, embed_dim=32, decoder_dim=32, mlp_ratio=4, n_heads=4, in_chans=2, out_chans=6)
    x = torch.randn(1, 2, img_size, img_size)
    o = model(x)
    print(o)
    print(o.shape)


    # img_size = 640
    # encoder = VisionTransformer(image_size=[img_size, img_size], patch_size=16, n_layers=4, d_model=96, d_ff=4, n_heads=4, n_cls=1, channels=2)
    # img = torch.randn(1, 2, img_size, img_size)
    # o = encoder(img, return_features=True)
    # print(o)
    # print(o.shape)
    # decoder = MaskTransformer(n_cls=10, patch_size=16, d_encoder=encoder.d_model, n_layers=4, n_heads=4, d_model=96, d_ff=4, drop_path_rate=0.0, dropout=0.0)
    # # Remove CLS token
    # o2 = decoder(o[:, 1:], [img_size, img_size])
    # print(o2)
    #
    # masks = F.interpolate(o2, size=(img_size, img_size), mode="bilinear")
    # print(masks.shape)
