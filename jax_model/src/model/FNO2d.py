"""
Modified from
https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_FNO_in_JAX.ipynb
"""

from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp


class SpectralConv2d(eqx.Module):
    real_weights1: jax.Array
    imag_weights1: jax.Array
    real_weights2: jax.Array
    imag_weights2: jax.Array
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.modes2 = modes

        scale = 1.0 / (in_channels * out_channels)

        real_key1, imag_key1, real_key2, imag_key2 = jax.random.split(key, 4)
        self.real_weights1 = jax.random.uniform(
            real_key1,
            (in_channels, out_channels, modes, modes),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights1 = jax.random.uniform(
            imag_key1,
            (in_channels, out_channels, modes, modes),
            minval=-scale,
            maxval=+scale,
        )
        self.real_weights2 = jax.random.uniform(
            real_key2,
            (in_channels, out_channels, modes, modes),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights2 = jax.random.uniform(
            imag_key2,
            (in_channels, out_channels, modes, modes),
            minval=-scale,
            maxval=+scale,
        )

    def complex_mult2d(
        self,
        x_hat,
        w,
    ):
        return jnp.einsum("ixy,ioxy->oxy", x_hat, w)

    def __call__(
        self,
        x,
    ):
        channels, X, T = x.shape

        x_hat = jnp.fft.fft2(
            x
        )  # since the input is complex, we use fft instead of rfft

        x_hat_under_modes1 = x_hat[:, : self.modes1, : self.modes2]
        x_hat_under_modes2 = x_hat[:, -self.modes1 :, -self.modes2 :]
        weights1 = self.real_weights1 + 1j * self.imag_weights1
        weights2 = self.real_weights2 + 1j * self.imag_weights2

        out_hat_under_modes1 = self.complex_mult2d(
            x_hat_under_modes1, weights1
        )
        out_hat_under_modes2 = self.complex_mult2d(
            x_hat_under_modes2, weights2
        )

        out_hat = jnp.zeros(
            (self.out_channels, x_hat.shape[-2], x_hat.shape[-1]),
            dtype=x_hat.dtype,
        )
        out_hat = out_hat.at[:, : self.modes1, : self.modes2].set(
            out_hat_under_modes1
        )
        out_hat = out_hat.at[:, -self.modes1 :, -self.modes2 :].set(
            out_hat_under_modes2
        )

        out = jnp.fft.ifft2(out_hat)

        return out


class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: eqx.nn.Conv2d
    activation: Callable

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        activation,
        *,
        key,
    ):
        spectral_conv_key, bypass_conv_key = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes,
            key=spectral_conv_key,
        )
        self.bypass_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            1,  # Kernel size is one, same as element-wise linear transformation
            key=bypass_conv_key,
        )
        self.activation = activation

    def __call__(
        self,
        x,
    ):
        spectral_out = self.spectral_conv(x)
        bypass_out = self.bypass_conv(x.real) + 1j * self.bypass_conv(x.imag)
        return self.activation(spectral_out + bypass_out)


class FNO2d(eqx.Module):
    lifting: eqx.nn.Conv2d
    fno_blocks: List[FNOBlock2d]
    projection: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        h_channels,
        activation,
        n_blocks=4,
        *,
        key,
    ):
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            h_channels,
            1,
            key=lifting_key,
        )

        self.fno_blocks = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)
            self.fno_blocks.append(
                FNOBlock2d(
                    h_channels,
                    h_channels,
                    modes,
                    activation,
                    key=subkey,
                )
            )

        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv2d(
            h_channels,
            out_channels,
            1,
            key=projection_key,
        )

    def __call__(
        self,
        x,
    ):
        x = self.lifting(x.real) + 1j * self.lifting(x.imag)

        for fno_block in self.fno_blocks:
            x = fno_block(x)

        x = self.projection(x.real) + 1j * self.projection(x.imag)

        return x

