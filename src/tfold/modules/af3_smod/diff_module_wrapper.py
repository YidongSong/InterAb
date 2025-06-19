"""The Wrapper of The AlphaFold3 structure module."""

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops import repeat

from tfold.modules.af3_smod.utils import log
from tfold.modules.af3_smod.utils import CentreRandomAugmentation


class DiffusionModuleWrapper(nn.Module):
    """Wrapper of DiffusionModule."""

    def __init__(
        self,
        net,  # protein sequence & structure denoising model (DiffusionModule)
        n_steps=200,  # number of sampling steps
        batch_size=48,   # number of noised structure
        sigma_min=0.0004,  # min noise level
        sigma_max=160,     # max noise level
        sigma_data=16,  # standard deviation of data distributions
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.5,    # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0=0.8,  # define in Algorithm 18
        gamma_min=1,  # define in Algorithm 18
        step_scale=1.5,  # define in Algorithm 18
        noise_scale=1.003,    # noise scale of sample
    ):
        super().__init__()
        """Constructor function."""

        self.net = net
        self.n_steps = n_steps
        self.batch_size = batch_size

        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.step_scale = step_scale
        self.noise_scale = noise_scale
        self.augmenter = CentreRandomAugmentation()

    def noise_distribution(self, batch_size, device):
        """Supplementary Page 24."""

        sigmas = (self.P_mean + self.P_std * torch.randn((batch_size,), device=device)).exp()
        sigmas = self.sigma_data * sigmas

        return sigmas

    def sample_schedule(self, n_steps=None):
        """sample schedule, described in Supplementary page 24"""

        n_steps = n_steps if n_steps is not None else self.n_steps
        inv_rho = 1 / self.rho
        steps = torch.arange(n_steps, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** inv_rho + steps / (n_steps - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)
        ) ** self.rho
        sigmas = self.sigma_data * sigmas
        sigmas = F.pad(sigmas, (0, 1), value=0.)   # last step is sigma value of 0.

        return sigmas

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma):  # Using by EDM, not in AF3
        return log(sigma/self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_pos,
        sigma,
        network_condition_kwargs,
    ):
        batch, device, dtype = noised_atom_pos.shape[0], noised_atom_pos.device, noised_atom_pos.dtype

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device, dtype=dtype)

        padded_sigmas = rearrange(sigma, 'b -> b 1 1')

        # Algorithm 20, line 2, follow the EDM paper.
        out = self.net(
            self.c_in(padded_sigmas) * noised_atom_pos,
            times=sigma,
            **network_condition_kwargs
        )

        # Algorithm 20, line 8
        out = self.c_skip(padded_sigmas) * noised_atom_pos + self.c_out(padded_sigmas) * out

        return out

    @torch.no_grad()
    def sample(
        self,
        atom_mask,
        n_steps=None,
        heun_sample=False,
        **network_condition_kwargs,
    ):
        """ Algorithm 18 Sample Diffusion."""
        self.net.enable_activation_checkpoint(False)

        n_steps = n_steps if n_steps is not None else self.n_steps
        atom_shape = (*atom_mask.shape, 3)   # [B, N, 3]

        sigmas = self.sample_schedule(n_steps)
        gammas = torch.where(sigmas >= self.gamma_min, self.gamma_0, 0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # line 1, atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_pos = init_sigma * torch.randn(
            atom_shape,
            device=atom_mask.device,
            dtype=self.net.tokens_to_atom_decoder_input_cond.weight.dtype
        )

        # gradually denoise
        for sigma, sigma_next, gamma in sigmas_and_gammas:

            # line 3
            atom_pos = self.augmenter(atom_pos)
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            # line 5 - line 7
            eps = self.noise_scale * torch.randn(atom_shape, dtype=atom_pos.dtype, device=atom_mask.device)
            sigma_hat = sigma * (gamma + 1)
            atom_pos_hat = atom_pos + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            # line 8
            model_output = self.preconditioned_network_forward(
                atom_pos_hat,
                sigma_hat,
                network_condition_kwargs=network_condition_kwargs
            )

            # line 9 - line 11
            denoised_over_sigma = (atom_pos - model_output) / sigma_hat
            atom_pos_next = atom_pos_hat + (sigma_next - sigma_hat) * denoised_over_sigma * self.step_scale

            # Apply 2nd order correction, which is not used in AF3 paper
            if heun_sample and sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    atom_pos_next,
                    sigma_next,
                    network_condition_kwargs=network_condition_kwargs,
                )
                denoised_prime_over_sigma = (atom_pos_next - model_output_next) / sigma_next
                atom_pos_next = atom_pos_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma) * self.step_scale

            atom_pos = atom_pos_next

        return atom_pos

    def forward(
        self,
        inputs,
        atom_inputs,
        sfea_tns,
        sfea_tns_trunk,
        pfea_tns_trunk,
        penc_tns,
        molecule_atom_lens
    ):
        self.net.enable_activation_checkpoint(self.training)

        sigmas = self.noise_distribution(self.batch_size, device=sfea_tns.device).to(sfea_tns.dtype)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1')
        amsk = inputs['base']['amsk']
        repeat_atom_inputs = {}

        def repeat_tensor(tensor):
            return repeat(tensor, 'b ... -> (b a) ...', a=self.batch_size)

        if self.batch_size > 1:
            (
                atom_pos,
                sfea_tns,
                sfea_tns_trunk,
                sfea_tns_trunk,
                pfea_tns_trunk,
                penc_tns,
                molecule_atom_lens,
                repeat_atom_inputs['atom_feats'],
                repeat_atom_inputs['atom_ref_pos'],
                repeat_atom_inputs['atom_ref_space_uid'],
                repeat_atom_inputs['molecule_atom_lens'],
                amsk,
            ) = tuple(
                repeat_tensor(t) for t in (
                    inputs['base']['atom'],
                    sfea_tns,
                    sfea_tns_trunk,
                    sfea_tns_trunk,
                    pfea_tns_trunk,
                    penc_tns,
                    molecule_atom_lens,
                    atom_inputs['atom_feats'],
                    atom_inputs['atom_ref_pos'],
                    atom_inputs['atom_ref_space_uid'],
                    atom_inputs['molecule_atom_lens'],
                    amsk,
                )
            )

        atom_pos = self.augmenter(atom_pos, mask=amsk)  # centre random augmentation
        noise = torch.randn_like(repeat_atom_inputs['atom_ref_pos'])
        noised_atom_pos = (atom_pos + padded_sigmas * noise).to(sfea_tns.dtype)

        denoised_atom_pos = self.preconditioned_network_forward(
            noised_atom_pos,
            sigmas,
            network_condition_kwargs=dict(
                atom_inputs=repeat_atom_inputs,
                sfea_tns=sfea_tns,
                sfea_tns_trunk=sfea_tns_trunk,
                pfea_tns_trunk=pfea_tns_trunk,
                penc_tns=penc_tns,
                molecule_atom_lens=molecule_atom_lens,
            )
        )

        return denoised_atom_pos, sigmas
