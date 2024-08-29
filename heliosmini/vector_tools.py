import jax
import optax


def build_masked(mask, data, shape=None):
    """
    Builds a masked array by setting elements from the data array at positions specified by the mask.

    Args:
        mask: A boolean array or an array of indices where data is to be placed.
        data: The data array to be inserted at positions specified by the mask.
        shape: Optional shape for the output array. If None, the shape will be derived from the mask and data.

    Returns:
        filled_array: An array with the specified shape, or the shape derived from the mask and data,
                      where elements from the data array are placed at positions specified by the mask,
                      and zeros elsewhere.
    """
    # Determine the shape of the empty array
    if shape is None:
        shape = jax.numpy.shape(mask) + jax.numpy.shape(data)[1:]
    # Use the mask to place data in the corresponding positions in the empty array
    filled_array = jax.numpy.zeros(shape).at[mask].set(data)

    return filled_array


def vector_field_interpolator(X, grid, epsilon):
    """
    Build a vector field by interpolating the norms and directions of X.

    Args:
        X: Initial vector field.
        grid: Grid points for interpolation.
        coords: Coordinates where interpolation is to be evaluated.
        epsilon: Small value to avoid division by zero in normalization.

    Returns:
        lambda: interpolator
    """
    # Normalize the initial light vector field
    phi, D = norm_vector(X, epsilon)

    # Create interpolators for the norm and direction
    i_phi = jax.scipy.interpolate.RegularGridInterpolator(grid, phi, fill_value = 0)
    i_D = jax.scipy.interpolate.RegularGridInterpolator(grid, D, fill_value = 0)

    # Re-normalize the interpolated direction and scale by the interpolated norm
    def interpolator(coords):
        return norm_vector(i_D(coords), epsilon)[1] * jax.numpy.expand_dims(
            i_phi(coords), axis=-1
        )
    
    return interpolator

def norm_vector(X, epsilon):
    """
    Normalize a vector X along the last axis.

    Args:
        X: Input vector.
        epsilon: Small value to avoid division by zero.

    Returns:
        norm: Norm of the input vector.
        direction: Normalized vector.
    """
    # Compute the norm of the vector X along the last axis
    norm = jax.numpy.linalg.norm(X, axis=-1)

    # Compute the direction (normalized vector) and avoid division by zero by adding epsilon
    direction = X * jax.numpy.expand_dims(jax.numpy.reciprocal(norm + epsilon), axis=-1)

    return norm, direction
