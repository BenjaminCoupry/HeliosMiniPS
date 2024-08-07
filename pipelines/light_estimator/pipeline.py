from pipelines.light_estimator import data_loading, data_processing, exports

def pipeline(parameters_path, data_path, out_path, seed):
    images_names, mask, N, I, first_image, meta_parameters, loading_time = data_loading.load_data(parameters_path,data_path)
    grid, rho_init, (L0,rho), losses, validity_mask, relative_grid, processing_times = data_processing.process_data(mask, N, I, meta_parameters, seed)
    preparation_time, first_estimation_time, gradient_descent_time = processing_times
    times = loading_time, preparation_time, first_estimation_time, gradient_descent_time
    exports.export(out_path, L0, N, I, rho, mask, rho_init, first_image, validity_mask, grid, losses, images_names, relative_grid, meta_parameters, times)
