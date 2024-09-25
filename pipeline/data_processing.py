import jax
import optax
import functools
import time
import tqdm

from heliosmini import least_squares
from heliosmini import model
from heliosmini import gradient
from heliosmini import grids
from heliosmini import vector_tools


def prepare_grid(mask, stride, meta_parameters):
    light_resolution = meta_parameters['model']['light_resolution']
    if light_resolution > 0:
        grid = grids.grid_over_mask(mask, light_resolution/stride)
    else:
        grid = tuple()
    return grid

def build_validity_mask(I, meta_parameters):
    I_grey = jax.numpy.mean(I,axis=1)
    validity_mask = jax.numpy.logical_and(I_grey>=meta_parameters['model']['validity_range']['min'],I_grey<=meta_parameters['model']['validity_range']['max'])
    return validity_mask

def preliminary_estimation(N, I, grid):
    nl, normdim = I.shape[-1], N.shape[-1]
    rho_init = jax.numpy.median(I,axis=-1)
    L_lstsq = least_squares.quadratic_light(rho_init,N,I)
    if len(grid) >0:
        L0_init = jax.numpy.zeros((jax.numpy.size(grid[0]),jax.numpy.size(grid[1]),nl,normdim)).at[...,:,:].set(L_lstsq)
    else:
        L0_init = L_lstsq
    return rho_init, L0_init

def get_batch_feeder(npix,batch_size):
    def feeder(parameters, feeder_state):
        key,feeder_state = jax.random.split(feeder_state)
        batch = jax.random.choice(key, npix, (batch_size,), replace=False)
        gradient_args = {'batch':batch}
        return gradient_args, feeder_state
    return feeder

def split_range(total_size, batch_size):
    nbatches = jax.numpy.maximum(1,total_size//batch_size)
    batches = jax.numpy.array_split(jax.numpy.arange(total_size), nbatches)
    return batches



def batched_loss(parameters, N, I, validity_mask, u_mask, v_mask, epsilon, batch, loss_function):
    (L0, rho, grid) = parameters
    lambertian_model = model.model(L0, rho[batch], grid, N[batch], u_mask[batch], v_mask[batch], epsilon, normalize=True)[0]
    loss_value = loss_function(predictions = lambertian_model, targets = I[batch], where = validity_mask[batch])
    return loss_value

def total_evaluation(L0, rho, grid, N, I, validity_mask, mask, meta_parameters):
    epsilon, (u_mask, v_mask), selection = meta_parameters['model']['epsilon'], jax.numpy.where(mask), jax.numpy.broadcast_to(validity_mask[:,None,:],I.shape)
    lambertian_model, Lmap = model.model(L0, rho, grid, N, u_mask, v_mask, epsilon)
    residuals = jax.numpy.abs(lambertian_model-I)
    total_error = jax.numpy.mean(residuals[selection]), jax.numpy.median(residuals[selection]), jax.numpy.std(residuals[selection])
    return residuals, lambertian_model, Lmap, total_error


def gradient_descent(L0_init, rho_init, grid_init, mask, N, I, validity_mask, meta_parameters):
    npix = I.shape[0]
    rng = jax.random.PRNGKey(meta_parameters['compute']['seed'])
    (u_mask, v_mask) = jax.numpy.where(mask)
    optimizer = optax.adam(meta_parameters['learning']['learning_rate'])
    mean_huber = lambda predictions, targets, where: jax.numpy.mean(optax.huber_loss(predictions = predictions, targets = targets, delta = meta_parameters['model']['delta']), where = where)
    batched_huber_loss = functools.partial(batched_loss, loss_function=mean_huber)
    kwargs = {'N':N, 'I':I, 'validity_mask':validity_mask[:,None,:], 'u_mask':u_mask, 'v_mask':v_mask, 'epsilon': meta_parameters['model']['epsilon']}
    feeder = get_batch_feeder(npix,min(npix,meta_parameters['learning']['batch_size']))
    with tqdm.tqdm(total=meta_parameters['learning']['steps'], desc='Descent (-.--e---)') as progress_bar:
        def callback(i, loss):
            if int(i)==0:
                progress_bar.reset()
            progress_bar.n = int(i)
            progress_bar.desc = f'Gradient Descent ({float(loss):.2e})'
            progress_bar.refresh()
        (L0, rho, grid), losses = gradient.gradient_descent(optimizer, batched_huber_loss, feeder, (L0_init, rho_init, grid_init), rng, meta_parameters['learning']['steps'],callback=callback, **kwargs)
    max_rho = jax.numpy.max(rho)
    L0_scaled = max_rho*model.normalize_L0(L0, meta_parameters['model']['epsilon'])
    rho_scaled = rho/max_rho
    return (L0_scaled, rho_scaled, grid), losses


 
def process_data(mask, N, I, stride, meta_parameters):
    t0 = time.time()
    grid_init = prepare_grid(mask, stride, meta_parameters)
    validity_mask = build_validity_mask(I, meta_parameters)
    t1 = time.time()
    rho_init, L0_init = preliminary_estimation(N, I, grid_init)
    t2 = time.time()
    (L0, rho, grid), losses = gradient_descent(L0_init, rho_init, grid_init, mask, N, I, validity_mask, meta_parameters)
    t3 = time.time()
    residuals, lambertian_model, Lmap, total_error = total_evaluation(L0, rho, grid, N, I, validity_mask, mask, meta_parameters)
    t4 = time.time()
    preparation_time, first_estimation_time, gradient_descent_time, mse_time = t1-t0,t2-t1,t3-t2, t4-t3
    processing_times = preparation_time, first_estimation_time, gradient_descent_time, mse_time
    return rho_init, (L0, rho, grid), losses, total_error, validity_mask, residuals, lambertian_model, Lmap, processing_times
