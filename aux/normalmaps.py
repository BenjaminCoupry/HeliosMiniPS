import jax

def r3_to_rgb(r3):
    rgb = 0.5 * (jax.numpy.clip(r3,-1,1) + 1)
    return rgb

def rgb_to_r3(rgb):
    r3 = (2.0 * jax.numpy.clip(rgb,0,1)) - 1.0
    return r3