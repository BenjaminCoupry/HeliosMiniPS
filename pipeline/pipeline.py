from pipeline import data_loading, data_processing, exports

def pipe(parameters_path, images_path, normals_path, mask_path, out_path):
    images_names, mask, N, I, first_image, meta_parameters, loading_time = data_loading.load_data(parameters_path,images_path, normals_path, mask_path)
    rho_init, (L0, rho, grid), losses, validity_mask, processing_times = data_processing.process_data(mask, N, I, meta_parameters)
    preparation_time, first_estimation_time, gradient_descent_time = processing_times
    times = loading_time, preparation_time, first_estimation_time, gradient_descent_time
    exports.export(out_path, L0, N, I, rho, mask, rho_init, first_image, validity_mask, grid, losses, images_names, meta_parameters, times)
