import numpy
from PIL import Image, ImageDraw, ImageOps

import matplotlib.pyplot as plt

def array_to_image(array):
    image_object = Image.fromarray(numpy.uint8(255.0 * array))
    return image_object

def image_to_array(image_object):
    array =  numpy.asarray(image_object)/ 255.0
    return array

def r3_to_rgb(r3):
    rgb = 0.5 * (numpy.clip(r3,-1,1) + 1)
    return rgb

def rgb_to_r3(rgb):
    r3 = (2.0 * numpy.clip(rgb,0,1)) - 1.0
    return r3

def to_mask(data):
    if numpy.ndim(data) == 2:
        return data>0
    else:
        return numpy.mean(data[:, :, :3], axis=-1) > 0.5

def plot_losses_with_sliding_mean(losses, filename):
    """
    Plots the losses with a sliding mean and saves the plot to a file.
    
    Parameters:
    losses (list or numpy array): Array of loss values.
    filename (str): The filename to save the plot.
    """
    # Convert losses to numpy array if it's not already
    losses = numpy.array(losses)
    
    # Calculate the sliding mean with a window of 10% of the data length
    window_size = max(1, int(len(losses) * 0.02))
    sliding_mean = numpy.convolve(losses, numpy.ones(window_size)/window_size, mode='valid')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the losses
    plt.plot(losses, label='Losses', color='blue')
    
    # Plot the sliding mean
    shift = int((window_size+1)/2)
    plt.plot(range(shift, sliding_mean.shape[0] + shift), sliding_mean, label=f'Sliding Mean (window = {window_size})', linestyle='--', color='orange')
    
    # Annotate the axes
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Losses and Sliding Mean Over Iterations')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()


def draw_grid(image, x_graduations, y_graduations):
    # Create a greyed-out version of the image
    new_image = image.convert('RGBA')
    # Create a drawing context
    draw = ImageDraw.Draw(new_image)
    # Get the dimensions of the image
    width, height = image.size
    w = int(min(width,height)*0.005)
    # Draw vertical lines
    for x in x_graduations:
        draw.line((x, 0, x, height), fill=(0, 0, 255), width=w)
    # Draw horizontal lines
    for y in y_graduations:
        draw.line((0, y, width, y), fill=(0, 0, 255), width=w)
    return new_image

def hash_image(image, binary_mask):
    hashed_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Get the dimensions of the image
    width, height = image.size

    # Create a drawing context for the mask
    draw = ImageDraw.Draw(hashed_mask)

    # Define the spacing and thickness of the hash lines
    spacing = int(min(width,height)*0.01)
    thickness = int(min(width,height)*0.002)

    # Draw the hash lines
    for x in range(0, width, spacing):
        draw.line((x, 0, x, height), fill=(255, 0, 0, 80), width=thickness)
    for y in range(0, height, spacing):
        draw.line((0, y, width, y), fill=(255, 0, 0, 80), width=thickness)

    hashed_mask = Image.composite(hashed_mask,Image.new('RGBA', image.size, (0, 0, 0, 0)),array_to_image(binary_mask).convert('L'))

    # Overlay the hashed mask onto the original image
    combined_image = Image.alpha_composite(image.convert('RGBA'), hashed_mask)
    return combined_image

def crop_mask(image, mask):
    u_mask, v_mask = numpy.where(mask)
    u_min,u_max,v_min,v_max = numpy.min(u_mask),numpy.max(u_mask),numpy.min(v_mask),numpy.max(v_mask)
    croped_image = image.crop((v_min,u_min,v_max,u_max))
    return croped_image

def add_grey(image):
    grey_overlay = Image.new('RGBA', image.size, (128, 128, 128, 128))
    grey_image = Image.alpha_composite(image.convert('RGBA'), grey_overlay)
    return grey_image


def stick_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with a width that is the sum of the widths of both images
    # and a height that is the maximum height of the two images
    new_width = width1 + width2
    new_height = max(height1, height2)
    combined_image = Image.new('RGB', (new_width, new_height))

    # Paste the first image at the left
    combined_image.paste(image1, (0, 0))

    # Paste the second image at the right
    combined_image.paste(image2, (width1, 0))
    return combined_image