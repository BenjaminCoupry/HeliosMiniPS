import glob
import time
import yaml
import os.path

from aux import IO
from aux import normalmaps

def load_data(parameters_path,data_path):
    t0 = time.time()
    images_files_list = glob.glob(os.path.join(data_path,'[0-9]*.png'))
    images_names = list(map(lambda p : os.path.basename(p),images_files_list))
    mask = IO.load_image(os.path.join(data_path,'mask.png'))>0
    N = normalmaps.rgb_to_r3(IO.load_image(os.path.join(data_path,'normals.png'),mask))
    I = IO.load_image_collection(images_files_list,mask)
    first_image = IO.load_image(images_files_list[0])
    with open(parameters_path, 'r') as file:
        meta_parameters = yaml.safe_load(file)
    t1 = time.time()
    loading_time = t1-t0
    return images_names, mask, N, I, first_image, meta_parameters, loading_time