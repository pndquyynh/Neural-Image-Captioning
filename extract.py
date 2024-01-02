from fontTools.ttLib import TTFont

# Load the font
font = TTFont('./fonts/NotoSansJP-Bold.ttf')

# Get the list of characters
chars = font.getBestCmap()

# Define the Unicode ranges for Japanese and Latin characters
latin_range = range(0x0020, 0x007F+1)
hiragana_range = range(0x3040, 0x309F+1)
katakana_range = range(0x30A0, 0x30FF+1)
kanji_range = range(0x4E00, 0x9FBF+1)

counter = 0
# Print the characters
for code, name in chars.items():
    if code in latin_range or code in hiragana_range or code in katakana_range or code in kanji_range:
        print(f"U+{code:04X}: {chr(code)} - {name}")
        counter += 1

print(counter)