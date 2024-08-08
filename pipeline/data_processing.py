import jax
import optax
import functools
import time

from heliosmini import least_squares
from heliosmini import model
from heliosmini import gradient
from heliosmini import grids


def prepare_grid(mask,meta_parameters):
    (nu, nv) = mask.shape
    grid = grids.grid_over_mask(mask,meta_parameters['model']['light_control_points'])
    relative_grid= grid[0]/nu, grid[1]/nv
    return grid, relative_grid

def build_validity_mask(I, meta_parameters):
    I_grey = jax.numpy.mean(I,axis=1)
    validity_mask = jax.numpy.logical_and(I_grey>=meta_parameters['model']['validity_range']['min'],I_grey<=meta_parameters['model']['validity_range']['max'])
    return validity_mask

def preliminary_estimation(N, I, grid):
    nl, normdim = I.shape[-1], N.shape[-1]
    rho_init = jax.numpy.median(I,axis=-1)
    L_lstsq = least_squares.quadratic_light(rho_init,N,I)
    L0_init = jax.numpy.zeros((jax.numpy.size(grid[0]),jax.numpy.size(grid[1]),nl,normdim)).at[...,:,:].set(L_lstsq)
    return rho_init, L0_init

def gradient_descent(L0_init, rho_init, mask, N, I, validity_mask, grid, meta_parameters):
    npix = I.shape[0]
    rng = jax.random.PRNGKey(meta_parameters['compute']['seed'])
    (u_mask, v_mask) = jax.numpy.where(mask)
    optimizer = optax.adam(meta_parameters['learning']['learning_rate'])
    kwargs = {'N':N, 'I':I, 'validity_mask':validity_mask[:,None,:],'grid':grid, 'u_mask':u_mask, 'v_mask':v_mask, 'epsilon': meta_parameters['model']['epsilon'], 'delta':meta_parameters['model']['delta']}
    partial_value_and_grad = functools.partial(model.stochastic_value_and_grad,npix= npix, batch_size=meta_parameters['learning']['batch_size'])
    (L0,rho), (rng,), losses = gradient.gradient_descent(optimizer, partial_value_and_grad, (L0_init,rho_init), (rng,), meta_parameters['learning']['steps'], **kwargs)
    return (L0,rho), losses 

 
def process_data(mask, N, I, meta_parameters):
    t0 = time.time()
    grid, relative_grid = prepare_grid(mask,meta_parameters)
    validity_mask = build_validity_mask(I, meta_parameters)
    t1 = time.time()
    rho_init, L0_init = preliminary_estimation(N, I, grid)
    t2 = time.time()
    (L0,rho), losses = gradient_descent(L0_init, rho_init, mask, N, I, validity_mask, grid, meta_parameters)
    t3 = time.time()
    preparation_time, first_estimation_time, gradient_descent_time = t1-t0,t2-t1,t3-t2
    processing_times = preparation_time, first_estimation_time, gradient_descent_time
    return grid, rho_init, (L0,rho), losses, validity_mask, relative_grid, processing_times
