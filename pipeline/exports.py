import os
import numpy
import jax

from aux import IO
from aux import plots
from aux import image_tools
from aux import normalmaps

from heliosmini import vector_tools




def export_result(out_path, L0, rho, grid, images_names, mask):
    os.makedirs(os.path.join(out_path,'result'), exist_ok=True)
    if len(grid) > 0:
        relative_grid_u, relative_grid_v = grid[0]/mask.shape[0], grid[1]/mask.shape[1]
        numpy.savez(os.path.join(out_path,'result','light_estimation.npz'), L0=L0,rho=rho,relative_grid_u = relative_grid_u, relative_grid_v = relative_grid_v, images_names = numpy.asarray(images_names,dtype=str))
    else:
        str_L0 = numpy.asarray(L0).astype(str)
        X = numpy.concatenate([numpy.asarray(images_names,dtype=str)[:,None],str_L0],axis=-1)
        numpy.savetxt(os.path.join(out_path,'result','light_estimation.lp'), X, fmt = '%s', header = str(len(images_names)), delimiter = ' ', comments='')

def export_diags(out_path, rho, rho_init, first_image, mask, validity_mask, residuals, grid, losses, total_error, times):
    loading_time, preparation_time, first_estimation_time, gradient_descent_time, mse_time = times
    os.makedirs(os.path.join(out_path,'diags'), exist_ok=True)
    residual_map = jax.numpy.mean(jax.numpy.mean(residuals,axis=-2),axis=-1,where=validity_mask)
    max_residual = jax.numpy.quantile(residual_map.at[jax.numpy.isnan(residual_map)].set(0),0.95)
    max_rho = jax.numpy.maximum(jax.numpy.max(rho),jax.numpy.max(rho_init))
    if len(grid) > 0:
        image_tools.draw_grid(image_tools.add_grey(IO.array_to_image(first_image),jax.numpy.logical_not(mask)),grid[1],grid[0]).save(os.path.join(out_path,'diags','grid.png'))
    else:
        image_tools.add_grey(IO.array_to_image(first_image),jax.numpy.logical_not(mask)).save(os.path.join(out_path,'diags','grid.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,jax.numpy.mean(validity_mask,axis=-1))),mask).save(os.path.join(out_path,'diags','validity.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,rho_init/max_rho)),mask).save(os.path.join(out_path,'diags','rho_init.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,rho/max_rho)),mask).save(os.path.join(out_path,'diags','rho_result.png'))
    plots.plot_losses_with_sliding_mean(losses,os.path.join(out_path,'diags','losses.png'))
    image_tools.crop_mask(IO.array_to_image(vector_tools.build_masked(mask,residual_map/max_residual)),mask).save(os.path.join(out_path,'diags','residual_map.png'))
    IO.print_to_text(os.path.join(out_path,'diags','times.txt'),['loading time         ',
                                                                'preparation time     ',
                                                                'first estimation time',
                                                                'gradient descent time',
                                                                'res computation time '],
                                                                [loading_time, preparation_time, first_estimation_time, gradient_descent_time, mse_time])
    IO.print_to_text(os.path.join(out_path,'diags','errors.txt'),['mean absolute error  ', 'median absolute error', 'standard deviation   '], total_error)


def export_lights_and_images(out_path, I, mask, validity_mask, residuals, lambertian_model, Lmap, images_names, meta_parameters):
    os.makedirs(os.path.join(out_path,'images'), exist_ok=True)
    os.makedirs(os.path.join(out_path,'lights'), exist_ok=True)
    os.makedirs(os.path.join(out_path,'residuals'), exist_ok=True)
    grey_residuals = jax.numpy.mean(residuals,axis=-2)
    flux, direction = vector_tools.norm_vector(Lmap,meta_parameters['model']['epsilon'])
    max_flux, max_residual = jax.numpy.max(flux), jax.numpy.quantile(grey_residuals[validity_mask],0.95)
    for i, file in enumerate(images_names):
        image_model = vector_tools.build_masked(mask,lambertian_model[:,:,i]*validity_mask[:,None,i])
        image_ref = vector_tools.build_masked(mask,I[:,:,i])
        validity = vector_tools.build_masked(mask,validity_mask[:,i])
        hashmask = jax.numpy.logical_and(jax.numpy.logical_not(validity),mask)
        image_direction = image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(vector_tools.build_masked(mask,normalmaps.r3_to_rgb(direction[...,i,:])*validity_mask[:,None,i])),hashmask),mask)
        image_flux = image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(vector_tools.build_masked(mask,numpy.clip(flux[...,i]/max_flux,0,1)*validity_mask[:,i])),hashmask),mask)
        image_tools.stick_images(image_direction,image_flux).save(os.path.join(out_path,'lights',file))
        croped_model = image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(image_model),hashmask),mask)
        croped_ref = image_tools.crop_mask(IO.array_to_image(image_ref),mask)
        image_tools.stick_images(croped_model,croped_ref).save(os.path.join(out_path,'images',file))
        image_tools.crop_mask(image_tools.hash_image(IO.array_to_image(vector_tools.build_masked(mask,validity_mask[:,i]*grey_residuals[:,i]/max_residual)),hashmask),mask).convert('RGB').save(os.path.join(out_path,'residuals',file))


def export(out_path, L0, I, rho, mask, rho_init, first_image, validity_mask, residuals, lambertian_model, Lmap, grid, losses, total_error, images_names, meta_parameters, times):
    export_result(out_path, L0, rho, grid, images_names, mask)
    export_diags(out_path, rho, rho_init, first_image, mask, validity_mask, residuals, grid, losses, total_error, times)
    export_lights_and_images(out_path,I, mask, validity_mask, residuals, lambertian_model, Lmap, images_names, meta_parameters)

