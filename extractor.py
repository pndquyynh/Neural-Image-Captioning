import os
import glob
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from PIL.Image import Transform
from tqdm import tqdm
from fontTools.ttLib import TTFont


def add_noise(img_input):
    # Convert PIL Image to numpy array
    img_arr = np.array(img_input)

    # Generate Gaussian noise
    noise = np.random.normal(0, 25, img_arr.shape)

    # Add the noise to the image
    noisy_img_array = img_arr + noise

    # Make sure the values are within the valid range (0-255)
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # Convert numpy array back to PIL Image
    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img

def random_augmentation(img):
    # Open the image file

    # List of transformations
    transformations = [
        ("rotate", lambda img: img.rotate(random.randint(-10, 10))),
        ("translate", lambda img: img.transform(img.size, Transform.AFFINE, (1, 0, random.randint(-5, 5), 0, 1, random.randint(-5, 5)))),
        ("brightness", lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))),
        ("contrast", lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 0.85))),
        ("sharpness", lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 0.85))),
        ("noise", lambda img: add_noise(img))
    ]

    # Apply a random transformation
    transform_name, transform_func = random.choice(transformations)
    return transform_func(img)



#char_fetched = TTFont('./fonts/NotoSansJP-Bold.ttf').getBestCmap()
font_files = glob.glob('./fonts/*.ttf')


# Define the Unicode ranges for Japanese and Latin characters
latin_range = list(range(0x0020, 0x007F+1))
hiragana_range = list(range(0x3040, 0x309F+1))
katakana_range = list(range(0x30A0, 0x30FF+1))
kanji_range = list(range(0x4E00, 0x9FBF+1))

common_characters = set(chr(code) for code in (latin_range + hiragana_range + katakana_range + kanji_range))

for font_file in font_files:
    char_fetched = TTFont(font_file).getBestCmap()
    characters_in_font = set(chr(code) for code,_ in char_fetched.items()
                             if code in latin_range or
                             code in hiragana_range or
                             code in katakana_range or
                             code in kanji_range)
    # Update common_characters to the intersection of itself and characters_in_font
    common_characters &= characters_in_font
    print(ImageFont.truetype(font_file).getname())
    print(os.path.basename(font_file)[:-4])



# Convert the set to a list
characters = list(common_characters)
print(characters)
print(len(characters))

for char in tqdm(characters, desc="Processing characters"):
    # Shuffle the list of fonts for each character
    random.shuffle(font_files)
    for font_file in font_files:
        # Load the font with a large size
        font = ImageFont.truetype(font_file, size=random.randint(40,50))

        # Create a new image with black background
        img = Image.new('L', (64, 64), color=0)

        d = ImageDraw.Draw(img)

        # Get the bounding box of the character
        # bbox = font.getbbox(char)

        # _, _, w, h = d.textbbox((0,0), char, font=font)
        d.text((32, 32), char, font=font, fill="white", anchor="mm")

        # Calculate the position to center the character
        # position = ((64 - bbox[2]) / 2, (64 - bbox[3] - bbox[1]) / 2)

        img_array = np.array(img)
        if np.all(img_array == 0):
            print("Font Error: ", char,"U+" + hex(ord(char))[2:].upper(),os.path.basename(font_file))
        else:
            #position = ((64 - bbox_width) / 2, (64 - bbox_height) / 2)


            # Draw the character in white
            # d.text(position, char, font=font, fill=255)

            # Decide whether this image goes into the training set or the validation set
            dataset_type = 'train' if font_files.index(font_file) < len(font_files) * 0.8 else 'valid'

            # Create the path to the output file
            unicode_char = "U+" + hex(ord(char))[2:].upper()
            out_dir = os.path.join('dataset', dataset_type, unicode_char)
            os.makedirs(out_dir, exist_ok=True)
            #out_path = os.path.join(out_dir, f'{unicode_char}_{font.getname()[0]+"_"+font.getname()[1]}.png')
            out_path = os.path.join(out_dir, f'{unicode_char}_{os.path.basename(font_file)[:-4]}.png')

            # Save the image
            img.save(out_path)

            if dataset_type == 'train':
                aug_img = random_augmentation(img)
                aug_out_path = os.path.join(out_dir, f'{unicode_char}_{os.path.basename(font_file)[:-4]}_aug.png')
                aug_img.save(aug_out_path)
