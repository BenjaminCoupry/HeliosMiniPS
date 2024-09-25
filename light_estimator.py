from pipeline import pipeline
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run the local light estimation pipeline.')
    parser.add_argument('--parameters_path', type=str, required=True, help='Path to the parameters.yaml file')
    parser.add_argument('--images_path', type=str, required=True, help='Path to the images (regex)')
    parser.add_argument('--normals_path', type=str, required=True, help='Path to the normalmap file')
    parser.add_argument('--mask_path', type=str, required=False, default=None, help='Path to the mask file')
    parser.add_argument('--black_path', type=str, required=False, default=None, help='Path to the black image')
    parser.add_argument('--out_path', type=str, required=True, help='Output path for the results')
    args = parser.parse_args()
    return args.parameters_path, args.images_path, args.normals_path, args.mask_path, args.black_path, args.out_path


def main():
    parameters_path, images_path, normals_path, mask_path, black_path, out_path = parse_args()
    pipeline.pipe(parameters_path, images_path, normals_path, mask_path, black_path, out_path)

if __name__ == '__main__':
    main()
