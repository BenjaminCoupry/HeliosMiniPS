import yaml
import matplotlib.pyplot as plt
import os
import glob
import numpy
import jax
import optax
import functools
import time

from PIL import Image

from aux import IO
from aux import normalmaps
from aux import image_tools

from heliosmini import vector_tools
from heliosmini import least_squares
from heliosmini import model
from heliosmini import gradient
from heliosmini import grids


#4e7 pixels (rgb)
parameters_path = '/home/bcoupry/Work/HeliosMiniPS/parameters.yaml'
data_path = '/media/bcoupry/T7 Shield/HeadMVPS/data/PS_DOME/msr_mini/'
out_path = '/media/bcoupry/T7 Shield/HeadMVPS/result/HeliosMini'

images_files_list = glob.glob(os.path.join(data_path,'[0-9]*.png'))


######################
t0 = time.time()
mask = IO.load_image(os.path.join(data_path,'mask.png'))>0
N = normalmaps.rgb_to_r3(IO.load_image(os.path.join(data_path,'normals.png',mask)))
I = IO.load_image_collection(images_files_list,mask)
first_image = IO.load_image(images_files_list[0])
with open(parameters_path, 'r') as file:
    meta_parameters = yaml.safe_load(file)
t1 = time.time()
loading_time = t1-t0


######################
t0 = time.time()
(npix,nc,nl) = I.shape

(u_mask, v_mask), (nu, nv) = jax.numpy.where(mask), mask.shape
grid = grids.grid_over_mask(mask,meta_parameters['model']['light_control_points'])
relative_grid_u, relative_grid_v = grid[0]/nu, grid[1]/nv

I_grey = jax.numpy.mean(I,axis=1)
validity_mask = jax.numpy.logical_and(I_grey>=meta_parameters['model']['validity_range']['min'],I_grey<=meta_parameters['model']['validity_range']['max'])
t1 = time.time()
preparation_time = t1-t0


######################
t0 = time.time()
rho_init = jax.numpy.median(I,axis=-1)
L_lstsq = least_squares.quadratic_light(rho_init,N,I)
L0_init = jax.numpy.zeros((jax.numpy.size(grid[0]),jax.numpy.size(grid[1]),nl,3)).at[...,:,:].set(L_lstsq)
t1 = time.time()
first_estimation_time = t1-t0


######################
t0 = time.time()
rng = jax.random.PRNGKey(0)
optimizer = optax.adam(meta_parameters['learning']['learning_rate'])
kwargs = {'N':N, 'I':I, 'validity_mask':validity_mask[:,None,:],'grid':grid, 'u_mask':u_mask, 'v_mask':v_mask, 'epsilon': meta_parameters['model']['epsilon'], 'delta':meta_parameters['model']['delta']}
partial_value_and_grad = functools.partial(model.stochastic_value_and_grad,npix= npix, batch_size=meta_parameters['learning']['batch_size'])
(L0,rho), (rng,), losses = gradient.gradient_descent(optimizer, partial_value_and_grad, (L0_init,rho_init), (rng,), meta_parameters['learning']['steps'], **kwargs)
t1 = time.time()
gradient_descent_time = t1-t0


######################
os.makedirs(os.path.join(out_path,'images'), exist_ok=True)
os.makedirs(os.path.join(out_path,'lights'), exist_ok=True)
os.makedirs(os.path.join(out_path,'diags'), exist_ok=True)
os.makedirs(os.path.join(out_path,'result'), exist_ok=True)

numpy.savez(os.path.join(out_path,'result','light_estimation.npz'), L0=L0,rho=rho,relative_grid_u = relative_grid_u, relative_grid_v = relative_grid_v)

max_rho = jax.numpy.maximum(jax.numpy.max(rho),jax.numpy.max(rho_init))
max_flux = jax.numpy.max(vector_tools.norm_vector(L0, meta_parameters['model']['epsilon'])[0])
IO.draw_grid(IO.add_grey(IO.array_to_image(first_image),jax.numpy.logical_not(mask)),grid[1],grid[0]).save(os.path.join(out_path,'diags','grid.png'))
IO.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,jax.numpy.mean(validity_mask,axis=-1))),mask).save(os.path.join(out_path,'diags','validity.png'))
IO.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,rho_init/max_rho)),mask).save(os.path.join(out_path,'diags','rho_init.png'))
IO.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,rho/max_rho)),mask).save(os.path.join(out_path,'diags','rho_result.png'))
IO.plot_losses_with_sliding_mean(losses,os.path.join(out_path,'diags','losses.png'))

IO.print_to_text(os.path.join(out_path,'diags','times.txt'),['loading time         ',
                                                             'preparation time     ',
                                                             'first estimation time',
                                                             'gradient descent time'],
                                                             [loading_time, preparation_time, first_estimation_time, gradient_descent_time])

for i, file in enumerate(map(lambda p : os.path.basename(p),images_files_list)):
    validity = vector_tools.build_masked(mask,validity_mask[:,i])
    hashmask = jax.numpy.logical_and(jax.numpy.logical_not(validity),mask)
    Lmap = vector_tools.vector_field_interpolator(L0[:,:,i,:],grid,meta_parameters['model']['epsilon'])((u_mask,v_mask))
    flux, direction = vector_tools.norm_vector(Lmap,meta_parameters['model']['epsilon'])
    image_direction = IO.crop_mask(IO.hash_image(IO.array_to_image(vector_tools.build_masked(mask,IO.r3_to_rgb(direction))),hashmask),mask)
    image_flux = IO.crop_mask(IO.hash_image(IO.array_to_image(vector_tools.build_masked(mask,numpy.clip(flux/max_flux,0,1))),hashmask),mask)
    IO.stick_images(image_direction,image_flux).save(os.path.join(out_path,'lights',file))
    image_model = vector_tools.build_masked(mask,model.rendering(rho,Lmap[:,None,:],N)[:,:,0])
    image_ref = vector_tools.build_masked(mask,I[:,:,i])
    IO.stick_images(IO.crop_mask(IO.hash_image(IO.array_to_image(image_model),hashmask),mask),IO.crop_mask(IO.array_to_image(image_ref),mask)).save(os.path.join(out_path,'images',file))


print()