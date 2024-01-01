import csv
import os
from PIL import Image, ImageDraw, ImageFont

# Path to the text file
text_file_path = r'F:\Current_Topics\Code\raw\raw'
background_image_path = r'F:\Current_Topics\Code\JESC\background.jpg'

# Read the Japanese text from the file
with open(text_file_path, 'r', encoding='utf-8') as file:
    japanese_text = file.read()

# Use PIL to create an image and insert the Japanese text into it
image = Image.new('RGBA', (500, 300), color = (73, 109, 137, 0))
d = ImageDraw.Draw(image)
# You may need to adjust the font and size depending on your system
font = ImageFont.truetype("arial.ttf", 15)
d.text((10,10), japanese_text, fill=(255, 255, 0), font=font)

# Load the background image
background = Image.open(background_image_path)

# Resize the background image to match the size of the text image
background = background.resize(image.size, Image.ANTIALIAS)

# Combine the text image with the background image
final_image = Image.alpha_composite(background.convert('RGBA'), image)

# Display the final image
final_image.show()

# Save the final image
final_image.save('text_image.png')