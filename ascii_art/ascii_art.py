from PIL import Image, ImageEnhance, ImageFilter
from reportlab.pdfgen import canvas
from reportlab.lib import pagesizes
import numpy as np
import sys


PAGESIZE_MAP = {
    "A0": pagesizes.A0,
    "A1": pagesizes.A0,
    "A2": pagesizes.A0,
    "A3": pagesizes.A0,
    "A4": pagesizes.A0,
    "A5": pagesizes.A0,
    "A6": pagesizes.A0,
    "A7": pagesizes.A0,
    "A8": pagesizes.A0,
    "A9": pagesizes.A0,
    "A10": pagesizes.A0,
}


def get_pagesize(arg):
    key = arg.upper()
    if key not in PAGESIZE_MAP:
        raise ValueError(f"Invalid page size: '{arg}'. Valid options are: {', '.join(PAGESIZE_MAP)}")
    return PAGESIZE_MAP[key]


def generate_ascii_art(filepath, max_width, max_height, rotate=True):
    aspect_ratio = 0.6
    punctuation = list("@$B%8WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

    with Image.open(filepath).convert("L") as image:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        image = ImageEnhance.Contrast(image).enhance(1.5)
        width, height = image.size
        if width > height and rotate:
            image = image.transpose(Image.ROTATE_90)
            width, height = image.size
        division_factor = max(width / max_width, height / max_height * aspect_ratio, 1)
        image = image.resize((int(width / division_factor), int(height / division_factor * aspect_ratio)))
        width, height = image.size
        brightness_array = np.array(image)
        section_length = 255 / len(punctuation)

        punctuation_matrix = []
        for i in range(height):
            row = []
            for j in range(width):
                index = int(brightness_array[i, j] // section_length)
                index = min(index, len(punctuation) - 1)  # prevent out-of-range
                row.append(punctuation[index])
            punctuation_matrix.append(row)

        ascii_art = "\n".join("".join(row) for row in punctuation_matrix)
    return ascii_art

def save_ascii_to_pdf(text, output_path, font_size, pagesize):
    page_width, page_height = pagesize
    c = canvas.Canvas(output_path, pagesize=pagesize)
    c.setFont("Courier", font_size)
    y = page_height - font_size
    for line in text.splitlines():
        c.drawString(0, y, line)
        y -= font_size
    c.save()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        pagesize = get_pagesize(str(sys.argv[1]))
        filepath = str(sys.argv[2])
        output_path = str(sys.argv[3])
        font_size = 2
    elif len(sys.argv) == 5:
        pagesize = get_pagesize(str(sys.argv[1]))
        filepath = str(sys.argv[2])
        output_path = str(sys.argv[3])
        font_size = int(sys.argv[4])
    else:
        pagesize = pagesizes.A4
        filepath = "subject2.png"
        output_path = "output.pdf"
        font_size = 2

    char_width = 0.6 * font_size
    char_height = font_size
    page_width, page_height = pagesize
    max_width = page_width // char_width
    max_height = page_height // char_height

    ascii_art = generate_ascii_art(filepath, max_width, max_height, False)
    save_ascii_to_pdf(ascii_art, output_path, font_size, pagesize)

