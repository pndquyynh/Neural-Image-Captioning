import csv
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def add_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 0.2, np_image.shape).astype(np.uint8)
    noisy_image = Image.fromarray(np_image + noise)
    return noisy_image

csv_file_path = 'ground_truth_eg.csv'
background_image_path = r'F:\Current_Topics\Code\JESC\background.jpg'
output_dir = r'F:\Current_Topics\Code\dataset_eg'
fonts_dir = r'F:\Current_Topics\Code\font'

font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf') and 'regular' not in f.lower()]
font_files_bold = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf') and 'bold' in f.lower()]

def random_augmentation(img, text, font, font_bold):
    transformed_image = Image.new('RGBA', img.size, color=(73, 109, 137, 0))
    d = ImageDraw.Draw(transformed_image)

    use_bold_font = [False]

    transformations = [
        ("brightness", lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))),
        ("contrast", lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 0.85))),
        ("sharpness", lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 0.85))),
        ("noise", lambda img: add_noise(img)),
        ("bold", lambda img: [use_bold_font.append(True), img][1]),  # Return the image as is
    ]

    transformation_name, transform = random.choice(transformations)
    transformed_image = transform(transformed_image)

    print(f"Applied {transformation_name} augmentation")  # Print the name of the applied transformation

    transformed_image = transformed_image.rotate(random.randint(-10, 10), expand=True)

    transformed_image = transformed_image.resize(img.size, Image.LANCZOS)

    if use_bold_font[-1]:
        font = font_bold

    d.text((20, 20), text, fill=(255, 255, 255), font=font)

    final_image = Image.alpha_composite(img, transformed_image)

    return final_image

with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for i, row in enumerate(reader):
        print(f"Row {i}: {row}")
        if len(row) < 2:
            continue

        japanese_text = row[1].replace(' ', '\u00A0')
        print(f"Japanese text: {japanese_text}")

        combined_text = japanese_text

        font_file = random.choice(font_files)
        font_path = os.path.join(fonts_dir, font_file)
        font = ImageFont.truetype(font_path, 45)

        font_file_bold = random.choice(font_files_bold)
        font_path_bold = os.path.join(fonts_dir, font_file_bold)
        font_bold = ImageFont.truetype(font_path_bold, 45)

        image = Image.new('RGBA', (1000, 64), color=(73, 109, 137, 0))
        d = ImageDraw.Draw(image)

        # Apply augmentation on even rows (0-indexed)
        if i % 2 == 0:
            d.text((0, 0), combined_text, fill=(255, 255, 255), font=font_bold)
            final_image = image
            final_image = random_augmentation(final_image, combined_text, font, font_bold)
            print("Applied augmentation")
        else:
            d.text((0, 0), combined_text, fill=(255, 255, 255), font=font)
            final_image = image
            print("Did not apply augmentation")

        bbox = final_image.getbbox()

        final_image = final_image.crop(bbox)

        background = Image.open(background_image_path)

        background = background.resize(final_image.size, Image.LANCZOS)

        final_image = Image.alpha_composite(background.convert('RGBA'), final_image)

        final_image.save(f'{output_dir}/image_{i}.png')