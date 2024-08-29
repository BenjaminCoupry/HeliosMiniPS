import os
import numpy
import jax

from aux import IO
from aux import plots
from aux import image_tools
from aux import normalmaps

from heliosmini import vector_tools
from heliosmini import model



def export_result(out_path, L0, rho, grid, images_names, mask):
    os.makedirs(os.path.join(out_path,'result'), exist_ok=True)
    if len(grid) > 0:
        relative_grid_u, relative_grid_v = grid[0]/mask.shape[0], grid[1]/mask.shape[1]
        numpy.savez(os.path.join(out_path,'result','light_estimation.npz'), L0=L0,rho=rho,relative_grid_u = relative_grid_u, relative_grid_v = relative_grid_v, images_names = numpy.asarray(images_names,dtype=str))
    else:
        str_L0 = numpy.asarray(L0).astype(str)
        X = numpy.concatenate([numpy.asarray(images_names,dtype=str)[:,None],str_L0],axis=-1)
        numpy.savetxt(os.path.join(out_path,'result','light_estimation.lp'), X, fmt = '%s', header = str(len(images_names)), delimiter = ' ', comments='')

def export_diags(out_path, rho, rho_init, first_image, mask, validity_mask, grid, losses, loading_time, preparation_time, first_estimation_time, gradient_descent_time):
    os.makedirs(os.path.join(out_path,'diags'), exist_ok=True)
    max_rho = jax.numpy.maximum(jax.numpy.max(rho),jax.numpy.max(rho_init))
    if len(grid) > 0:
        image_tools.draw_grid(image_tools.add_grey(IO.array_to_image(first_image),jax.numpy.logical_not(mask)),grid[1],grid[0]).save(os.path.join(out_path,'diags','grid.png'))
    else:
        image_tools.add_grey(IO.array_to_image(first_image),jax.numpy.logical_not(mask)).save(os.path.join(out_path,'diags','grid.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,jax.numpy.mean(validity_mask,axis=-1))),mask).save(os.path.join(out_path,'diags','validity.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,rho_init/max_rho)),mask).save(os.path.join(out_path,'diags','rho_init.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,rho/max_rho)),mask).save(os.path.join(out_path,'diags','rho_result.png'))
    plots.plot_losses_with_sliding_mean(losses,os.path.join(out_path,'diags','losses.png'))
    IO.print_to_text(os.path.join(out_path,'diags','times.txt'),['loading time         ',
                                                                'preparation time     ',
                                                                'first estimation time',
                                                                'gradient descent time'],
                                                                [loading_time, preparation_time, first_estimation_time, gradient_descent_time])

def export_lights_and_images(out_path, rho, N,I, mask, validity_mask, images_names, grid, L0, meta_parameters):
    os.makedirs(os.path.join(out_path,'images'), exist_ok=True)
    os.makedirs(os.path.join(out_path,'lights'), exist_ok=True)
    (u_mask, v_mask) = jax.numpy.where(mask)
    max_flux = jax.numpy.max(vector_tools.norm_vector(L0, meta_parameters['model']['epsilon'])[0])
    if len(grid) == 0:
        L0 = L0[None,...]
    for i, file in enumerate(images_names):
        Lmap = vector_tools.vector_field_interpolator(L0[...,i,:],grid,meta_parameters['model']['epsilon'])((u_mask,v_mask))
        flux, direction = vector_tools.norm_vector(Lmap,meta_parameters['model']['epsilon'])
        image_model = vector_tools.build_masked(mask,model.rendering(rho,Lmap[:,None,:],N)[:,:,0])
        image_ref = vector_tools.build_masked(mask,I[:,:,i])
        validity = vector_tools.build_masked(mask,validity_mask[:,i])
        hashmask = jax.numpy.logical_and(jax.numpy.logical_not(validity),mask)
        image_direction = image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(vector_tools.build_masked(mask,normalmaps.r3_to_rgb(direction))),hashmask),mask)
        image_flux = image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(vector_tools.build_masked(mask,numpy.clip(flux/max_flux,0,1))),hashmask),mask)
        image_tools.stick_images(image_direction,image_flux).save(os.path.join(out_path,'lights',file))
        croped_model = image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(image_model),hashmask),mask)
        croped_ref = image_tools.crop_mask(IO.array_to_image(image_ref),mask)
        image_tools.stick_images(croped_model,croped_ref).save(os.path.join(out_path,'images',file))

def export(out_path, L0, N, I, rho, mask, rho_init, first_image, validity_mask, grid, losses, images_names, meta_parameters, times):
    loading_time, preparation_time, first_estimation_time, gradient_descent_time = times
    export_result(out_path, L0, rho, grid, images_names, mask)
    export_diags(out_path, rho, rho_init, first_image, mask, validity_mask, grid, losses, loading_time, preparation_time, first_estimation_time, gradient_descent_time)
    export_lights_and_images(out_path, rho, N,I, mask, validity_mask, images_names, grid, L0, meta_parameters)

