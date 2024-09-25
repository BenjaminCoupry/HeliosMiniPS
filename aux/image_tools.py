from PIL import Image, ImageDraw
import numpy


from aux import IO


def draw_grid(image, x_graduations, y_graduations):
    # Create a greyed-out version of the image
    new_image = image.convert('RGBA')
    # Create a drawing context
    draw = ImageDraw.Draw(new_image)
    # Get the dimensions of the image
    width, height = image.size
    w = max(1,int(min(width,height)*0.02))
    for x in x_graduations:
        for y in y_graduations:
            draw.ellipse((x-w, y-w, x+w, y+w), fill=(0, 0, 255), outline=(0, 0, 200))
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

    hashed_mask = Image.composite(hashed_mask,Image.new('RGBA', image.size, (0, 0, 0, 0)),IO.array_to_image(binary_mask).convert('L'))

    # Overlay the hashed mask onto the original image
    combined_image = Image.alpha_composite(image.convert('RGBA'), hashed_mask)
    return combined_image

def crop_mask(image, mask):
    u_mask, v_mask = numpy.where(mask)
    u_min,u_max,v_min,v_max = numpy.min(u_mask),numpy.max(u_mask),numpy.min(v_mask),numpy.max(v_mask)
    croped_image = image.crop((v_min,u_min,v_max,u_max))
    return croped_image

def add_grey(image, binary_mask):
    grey_overlay = Image.new('RGBA', image.size, (128, 128, 128, 128))
    transparent_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay = Image.composite(grey_overlay,transparent_overlay,IO.array_to_image(binary_mask).convert('L'))
    grey_image = Image.alpha_composite(image.convert('RGBA'), overlay)
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