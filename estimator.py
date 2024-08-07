from pipelines.light_estimator import pipeline

#4e7 pixels (rgb)
parameters_path = '/home/bcoupry/Work/HeliosMiniPS/parameters.yaml'
data_path = '/media/bcoupry/T7 Shield/HeadMVPS/data/PS_DOME/msr_mini/'
out_path = '/media/bcoupry/T7 Shield/HeadMVPS/result/HeliosMini'
seed = 0
pipeline.pipeline(parameters_path,data_path,out_path, seed)