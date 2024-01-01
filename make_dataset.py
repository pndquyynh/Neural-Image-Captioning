import csv
import os
import random
from PIL import Image, ImageDraw, ImageFont

# Path to the CSV file
csv_file_path = 'ground_truth_train.csv'
background_image_path = r'F:\Current_Topics\Code\JESC\background.jpg'
output_dir = r'F:\Current_Topics\Code\dataset\train'
fonts_dir = r'F:\Current_Topics\Code\font'  # directory containing your fonts

# Get a list of all font files in the fonts directory
font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf')]

csv.field_size_limit(100000000)  # Set a sufficiently large limit

# Open the CSV file and read each line
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        # Print the row to the console for debugging
        print(f"Row {i}: {row}")

        # Skip rows that don't have at least two columns (English and Japanese sentences)
        if len(row) < 2:
            continue

        # Get the English and Japanese text from the row, replace spaces with a non-breaking space, and add a newline after each line
        english_text = row[0].replace(' ', '\u00A0') + '\n'
        japanese_text = row[1].replace(' ', '\u00A0') + '\n\n'  # add an additional newline to create a blank line

        # Print the English and Japanese text to the console for debugging
        print(f"English text: {english_text}")
        print(f"Japanese text: {japanese_text}")

        # Combine English and Japanese text
        combined_text = english_text + japanese_text

        # Use PIL to create an image and insert the combined text into it
        image = Image.new('RGBA', (500, 300), color = (73, 109, 137, 0))
        d = ImageDraw.Draw(image)

        # Randomly select a font
        font_file = random.choice(font_files)
        font_path = os.path.join(fonts_dir, font_file)

        # Append '-Bold' to the font name
        bold_font_path = font_path.replace('.ttf', '-Bold.ttf')

        try:
            # Try to use the bold font to draw the text
            font = ImageFont.truetype(bold_font_path, 15)
            d.text((10,10), combined_text, fill=(255, 255, 255), font=font)  # Change fill to the RGB color of your choice
        except IOError:
            # If the bold font doesn't exist or doesn't support the characters in the text, fall back to the regular font
            font = ImageFont.truetype(font_path, 15)
            d.text((10,10), combined_text, fill=(255, 255, 255), font=font)

        # Load the background image
        background = Image.open(background_image_path)

        # Resize the background image to match the size of the text image
        background = background.resize(image.size, Image.LANCZOS)

        # Combine the text image with the background image
        final_image = Image.alpha_composite(background.convert('RGBA'), image)

        # Save the final image
        final_image.save(f'{output_dir}/image_{i}.png')