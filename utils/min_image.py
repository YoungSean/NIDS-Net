from PIL import Image

# Open the original image
img_name = "test_031.jpg"
image_path = 'test_data/test_1/'+img_name  # Replace with your image path
image = Image.open(image_path)

# Calculate the new dimensions
new_width = image.width // 4
new_height = image.height // 4

# Resize the image
smaller_image = image.resize((new_width, new_height), Image.LANCZOS)

# Show the smaller image
smaller_image.show()

# Or save the smaller image
smaller_image.save('test_data/test_4/'+img_name)