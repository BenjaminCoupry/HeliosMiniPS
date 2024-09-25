from pipeline import data_loading, data_processing, exports

def pipe(parameters_path, images_path, normals_path, mask_path, black_path, out_path):
    images_names, mask, N, I, first_image, meta_parameters, loading_time = data_loading.load_data(parameters_path, images_path, normals_path, mask_path, black_path)
    rho_init, (L0, rho, grid), losses, total_error, validity_mask, residuals, lambertian_model, Lmap, processing_times = data_processing.process_data(mask, N, I, meta_parameters)
    times = (loading_time,) + processing_times
    exports.export(out_path, L0, I, rho, mask, rho_init, first_image, validity_mask, residuals, lambertian_model, Lmap, grid, losses, total_error, images_names, meta_parameters, times)
