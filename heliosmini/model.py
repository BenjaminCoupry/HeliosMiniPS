import jax
import optax
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

def normalize_L0(L0, epsilon):
    mean_norm = jax.numpy.mean(vector_tools.norm_vector(L0, epsilon)[0],axis=-1)
    normalized_L0 = L0/mean_norm[...,None,None]
    return normalized_L0

def get_light_map(L0, grid, u_mask, v_mask, epsilon, normalize=False):
    if normalize:
        normalized_L0 = normalize_L0(L0, epsilon)
    else:
        normalized_L0 = L0
    if len(grid) > 0:
        Lmap = vector_tools.vector_field_interpolator(normalized_L0,grid,epsilon)((u_mask,v_mask))
    else:
        Lmap = normalized_L0
    return Lmap


def model(L0, rho, grid, N, u_mask, v_mask, epsilon, normalize=False):
    Lmap = get_light_map(L0, grid, u_mask, v_mask, epsilon, normalize=normalize)
    lambertian_model = rendering(rho, Lmap, N)
    return lambertian_model, Lmap


