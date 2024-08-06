import jax

def grid_over_mask(mask, npoints):
    u_mask, v_mask = jax.numpy.where(mask)
    u_min,u_max,v_min,v_max = jax.numpy.min(u_mask),jax.numpy.max(u_mask),jax.numpy.min(v_mask),jax.numpy.max(v_mask)
    nu, nv = u_max-u_min+1, v_max-v_min+1
    ratio = nu/nv 
    nlv = int(jax.numpy.sqrt(npoints/ratio))
    nlu = int(jax.numpy.sqrt(npoints*ratio))
    grid = jax.numpy.linspace(u_min,u_max,nlu),jax.numpy.linspace(v_min,v_max,nlv)
    return grid