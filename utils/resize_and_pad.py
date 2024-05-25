from PIL import Image

def resize_and_pad(image_path, final_size=(448, 448), padding_color_image=(255, 255, 255), padding_color_mask=0):
    # Load the image or mask
    img = Image.open(image_path)

    # Calculate the resize target maintaining aspect ratio
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1:  # Width is greater than height
        new_width = final_size[0]
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = final_size[1]
        new_width = round(new_height * aspect_ratio)

    # Resize the image or mask
    img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate padding sizes
    padding_left = (final_size[0] - new_width) // 2
    padding_top = (final_size[1] - new_height) // 2
    padding_right = final_size[0] - new_width - padding_left
    padding_bottom = final_size[1] - new_height - padding_top

    # Determine padding color (white for images, black for masks)
    if img.mode == "L":  # Grayscale, likely a mask
        padding_color = padding_color_mask
    else:
        padding_color = padding_color_image

    # Create a new image with the specified dimensions and padding color
    img_padded = Image.new(img.mode, final_size, padding_color)
    img_padded.paste(img_resized, (padding_left, padding_top))

    # show the padded image
    img.show()
    img_padded.show()

# Paths to your image and mask
image_path = 'database/Objects/000_aveda_shampoo/images/001.jpg'
mask_path = 'database/Objects/000_aveda_shampoo/masks/001.png'

# Output paths for the resized and padded image and mask
output_image_path = 'path/to/your/resized_padded_image.jpg'
output_mask_path = 'path/to/your/resized_padded_mask.jpg'

# Resize and pad the image
resize_and_pad(image_path, final_size=(448, 448), padding_color_image=(255, 255, 255))

# Convert the mask to grayscale, resize, and pad it
resize_and_pad(mask_path, final_size=(448, 448), padding_color_mask=0)
