import numpy
from PIL import Image
import tqdm

def array_to_image(array, bit_depth = 8):
    sacled_array = (numpy.power(2,bit_depth) - 1) * numpy.clip(array,0,1)
    if bit_depth == 8:
        image_object = Image.fromarray(numpy.uint8(sacled_array))
    if bit_depth == 16:
        image_object = Image.fromarray(numpy.uint16(sacled_array))
    return image_object

def image_to_array(image_object):
    array =  numpy.asarray(image_object)
    if array.dtype == numpy.uint8:
        bit_depth = 8
    elif array.dtype == numpy.uint16:
        bit_depth = 16
    scaled_array = array/(numpy.power(2,bit_depth) - 1)
    return scaled_array

def load_image(path, stride=1 ,mask = ...):
    with Image.open(path) as image:
        array = image_to_array(image)[::stride,::stride][mask]
    return array

def load_image_collection(paths, stride=1, mask = ...):
    loaded_list = []
    for path in tqdm.tqdm(paths,desc = 'Loading Images'):
        array = load_image(path, stride, mask)
        loaded_list.append(array)
    arrays = numpy.stack(loaded_list,axis=-1)
    return arrays

def print_to_text(path,keys,values):
    with open(path, "w") as file:
        for key,value in zip(keys,values):
            file.write("{}: {}\n".format(key,value))

