import jax

def quadratic_light(rho, N, I):
    flat_I = jax.numpy.reshape(I, (-1, jax.numpy.shape(I)[-1]))
    M = jax.numpy.reshape(
        jax.numpy.einsum("...c,...k->...ck", rho, N), (-1, jax.numpy.shape(N)[-1])
    )
    L = jax.numpy.linalg.lstsq(M,flat_I)[0].T
    return L

