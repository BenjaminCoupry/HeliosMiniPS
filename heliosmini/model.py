import jax

from heliosmini import vector_tools


def rendering(rho, L, N):
    """
    Render the image using a simple shading model.

    Args:
        rho: Reflectance map.
        L: Light map.
        N: Normal map.

    Returns:
        render: Rendered image.
    """
    render = jax.nn.relu(
        jax.numpy.einsum("...c, ...lk, ...k -> ...cl", jax.nn.relu(rho), L, N)
    )
    return render

def stochastic_value_and_grad(parameters, refreshed_parameters, N, I, validity_mask, grid, u_mask, v_mask, npix, epsilon, delta, batch_size):
    def loss(parameters, N, I, validity_mask, grid, u_mask, v_mask, epsilon, delta):
        (L0, rho) = parameters
        Lmap = vector_tools.vector_field_interpolator(L0,grid,epsilon)((u_mask[batch],v_mask[batch]))
        lambertian_model = rendering(rho[batch], Lmap, N[batch])
        value = vector_tools.masked_huber_loss(lambertian_model,I[batch],delta,validity_mask[batch])
        return value
    (rng,) = refreshed_parameters
    key,rng = jax.random.split(rng)
    batch = jax.random.choice(key, npix, (batch_size,), replace=False)
    value, grad = jax.value_and_grad(loss)(parameters, N, I, validity_mask, grid, u_mask, v_mask, epsilon, delta)
    return value, grad, (rng,)