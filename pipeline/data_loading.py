import glob
import time
import yaml
import os.path
import numpy


from aux import IO
from aux import normalmaps

def compute_stride(max_pixels, nb_images, npix):
    if max_pixels==None:
        return 1
    else:
        stride = max(1,int(numpy.ceil(numpy.sqrt(npix*nb_images/max_pixels))))
        return stride
    
def get_mask(normals_path, mask_path):
    if mask_path is not None:
        mask = IO.load_image(glob.glob(mask_path)[0])>0
    else:
        normalmap = normalmaps.rgb_to_r3(IO.load_image(glob.glob(normals_path)[0]))
        norm = numpy.linalg.norm(normalmap,axis=-1)
        mask = numpy.logical_and(norm>0.98, norm<1.02)
    return mask


def load_data(parameters_path, images_path, normals_path, mask_path, black_path):
    t0 = time.time()
    with open(parameters_path, 'r') as file:
        meta_parameters = yaml.safe_load(file)
    images_files_list = sorted(glob.glob(images_path))
    images_names = list(map(lambda p : os.path.basename(p),images_files_list))
    mask = get_mask(normals_path, mask_path)
    stride = compute_stride(meta_parameters['compute']['max_pixels'], len(images_files_list), numpy.count_nonzero(mask))
    mask = mask[::stride,::stride]
    first_image = IO.load_image(images_files_list[0],stride=stride)
    N = normalmaps.rgb_to_r3(IO.load_image(glob.glob(normals_path)[0],mask=mask, stride=stride))
    if black_path is not None:
        black = IO.load_image(glob.glob(black_path)[0],mask=mask, stride=stride)
    else:
        black = 0
    I_abs = IO.load_image_collection(images_files_list, mask=mask, stride=stride)
    I = numpy.maximum(0, I_abs-black)
    t1 = time.time()
    loading_time = t1-t0
    return images_names, mask, N, I, first_image, meta_parameters, loading_time