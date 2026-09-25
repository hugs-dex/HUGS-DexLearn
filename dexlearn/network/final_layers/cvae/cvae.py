import torch
from torch import nn

from ..mlp import BasicMLP


class VAE(nn.Module):

    def __init__(
        self,
        encoder_layer_sizes,
        latent_size,
        decoder_layer_sizes,
        conditional=True,
        condition_size=1024,
    ):
        super().__init__()

        if conditional:
            assert condition_size > 0

        # assert type(encoder_layer_sizes) == list
        # assert type(latent_size) == int
        # assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_size
        )
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_size
        )

    def forward(self, x, c=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        # print('encoder', self.MLP)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)  # [B, 30+1024]
        # print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        # print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        # print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        # print('decoder', self.MLP)

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            # print('z size {}'.format(z.size()))

        x = self.MLP(z)

        return x


class GraspCVAE(nn.Module):
    def __init__(
        self,
        config,
        obj_inchannel=3,
        cvae_encoder_sizes=[1024, 512, 256],
        cvae_latent_size=64,
        cvae_decoder_sizes=[1024, 256],
        cvae_condition_size=512,
    ):
        super(GraspCVAE, self).__init__()
        self.config = config

        self.obj_inchannel = obj_inchannel
        self.cvae_encoder_sizes = cvae_encoder_sizes
        self.cvae_latent_size = cvae_latent_size
        self.cvae_decoder_sizes = cvae_decoder_sizes
        self.cvae_condition_size = cvae_condition_size
        self.cvae_encoder_sizes[0] = cvae_condition_size
        self.cvae_decoder_sizes[0] = cvae_condition_size
        latent_num = 12
        self.cvae_decoder_sizes.append(latent_num)

        self.hand_encoder = BasicMLP(latent_num, config.feature_dim)

        self.cvae = VAE(
            encoder_layer_sizes=self.cvae_encoder_sizes,
            latent_size=self.cvae_latent_size,
            decoder_layer_sizes=self.cvae_decoder_sizes,
            condition_size=self.cvae_condition_size,
        )

    def forward(self, obj_glb_feature, pose=None, inference=False):
        """
        :param obj_pc: [B, 3+n, N1]
        :param hand_xyz: [B, 3, N2]
        :return: reconstructed hand params
        """

        B = obj_glb_feature.size(0)

        if not inference:
            hand_input = torch.cat([pose["trans"], pose["rot"].reshape(-1, 9)], dim=-1)
            hand_glb_feature = self.hand_encoder(hand_input)
            recon, means, log_var, z = self.cvae(
                hand_glb_feature, obj_glb_feature
            )  # recon: [B, 30] or [B, 28]
            loss = self.cal_loss(hand_input, recon, means, log_var)
            return loss
        else:
            recon = self.cvae.inference(B, obj_glb_feature)
            return recon[..., :3], recon[..., 3:12]

    def cal_loss(self, hand_gt, hand_pred, mean, log_var):

        loss_trans = (hand_pred[:, :3] - hand_gt[:, :3]).abs().mean()
        loss_rot = (hand_pred[:, 3:12] - hand_gt[:, 3:12]).abs().mean()
        loss_KLD = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum()

        return loss_trans, loss_rot, loss_KLD
