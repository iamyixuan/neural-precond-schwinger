import jax
import equinox as eqx
import jax.numpy as jnp


def count_trainable_params(model) -> int:
    # Keep only array leaves (i.e. trainable stuff, assuming you used static_field/field(static=True) correctly)
    trainable, _ = eqx.partition(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(trainable)
    return sum(x.size for x in leaves)


class ComplexConv2d(eqx.Module):
    conv_layer: eqx.nn.Conv2d

    def __init__(self, *args, **kwargs):
        self.conv_layer = eqx.nn.Conv2d(*args, **kwargs)

    def __call__(self, x):
        x_real = x.real
        x_imag = x.imag
        return self.conv_layer(x_real) + 1j * self.conv_layer(x_imag)


class PrecondCNN(eqx.Module):
    conv_layers: list

    def __init__(
        self,
        inch,
        outch,
        activation,
        kernel_size,
        n_layers,
        hidden_dim,
        key,
    ):
        keys = jax.random.split(key, n_layers)
        padding = int((kernel_size - 1) / 2)
        self.conv_layers = []
        for i in range(n_layers):
            if i == 0:
                self.conv_layers.append(
                    ComplexConv2d(
                        inch,
                        hidden_dim,
                        kernel_size,
                        padding=padding,
                        key=keys[i],
                    )
                )
            else:
                self.conv_layers.append(
                    ComplexConv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        padding=padding,
                        key=keys[i],
                    )
                )
            if i < n_layers - 1:
                self.conv_layers.append(activation)
            else:
                self.conv_layers.append(
                    ComplexConv2d(
                        hidden_dim,
                        outch,
                        kernel_size,
                        padding=padding,
                        key=keys[i],
                    )
                )

    def __call__(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        return x
