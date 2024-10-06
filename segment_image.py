from PIL import Image, ImageDraw, ImageFont

def segment_image(input_image_path, output_image_path):
    image = Image.open(input_image_path)
    draw = ImageDraw.Draw(image)

    segment_size = 64
    width, height = image.size

    font = ImageFont.load_default()

    segment_number = 0

    for y in range(0, height, segment_size):
        for x in range(0, width, segment_size):
            draw.rectangle([x, y, x + segment_size, y + segment_size], outline="black", width=1)

            text_position = (x + 5, y + 5)
            draw.text(text_position, str(segment_number), fill="black", font=font)

            segment_number += 1

    image.save(output_image_path)

input_image = "images/I23.BMP"
output_image = "images/I23-segmented.BMP"
segment_image(input_image, output_image)
