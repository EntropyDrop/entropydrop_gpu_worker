import os
from PIL import Image

def remove_bg(img):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mask = Image.open(os.path.join(current_dir, 'skin-mask.png'))
    decor_mask = Image.open(os.path.join(current_dir, 'skin-decor-mask.png'))
    img = img.convert('RGBA')
    bg_color = img.getpixel((0, 0))

    def color_diff(a, b):
        return 0.299*(a[0]-b[0])**2 + 0.587*(a[1]-b[1])**2 + 0.114*(a[2]-b[2])**2

    dot_color = (255, 255, 255)
    ratio = img.width // 64
    ignore_map = {}
    t1 = 3000
    t2 = 2000
    def safe_getpixel(px, py):
        return img.getpixel((min(px, img.width - 1), min(py, img.height - 1)))

    for x in range(64):
        for y in range(64):
            if mask.getpixel((x, y))[3] == 0:
                if (color_diff(safe_getpixel(x*ratio+2, y*ratio+2),  dot_color) < t1 or\
                    color_diff(safe_getpixel(x*ratio+2, y*ratio+3), dot_color) < t1 or\
                    color_diff(safe_getpixel(x*ratio+3, y*ratio+3), dot_color) < t1 or\
                    color_diff(safe_getpixel(x*ratio+3, y*ratio+2), dot_color) < t1) and\
                    (color_diff(safe_getpixel(x*ratio+1, y*ratio+1), bg_color) < t2 or\
                    color_diff(safe_getpixel(x*ratio+4, y*ratio+4), bg_color) < t2 or\
                    color_diff(safe_getpixel(x*ratio+1, y*ratio+4), bg_color) < t2 or\
                    color_diff(safe_getpixel(x*ratio+4, y*ratio+1), bg_color) < t2):
                    ignore_map[(x, y)] = True
                else:
                    # Set the color of the 6x6 grid based on the center 4x4 pixels
                    c = safe_getpixel(x*ratio+2, y*ratio+2)
                    for i in range(x*ratio, min(x*ratio+6, img.width)):
                        for j in range(y*ratio, min(y*ratio+6, img.height)):
                            img.putpixel((i, j), c)

    img = img.resize((64, 64), Image.BOX)
    for x in range(64):
        for y in range(64):
            if ignore_map.get((x, y), False):
                img.putpixel((x, y), (0, 0, 0, 0))
            elif decor_mask.getpixel((x, y))[3] == 0 and mask.getpixel((x, y))[3] == 0:
                img.putpixel((x, y), (0, 0, 0, 0))
    return img
