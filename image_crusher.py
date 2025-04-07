import os
from PIL import Image, ImageOps

to_crush = os.listdir('image/Crusher')
to_crush.remove('.gitignore')

for run_image in to_crush:
    with Image.open("image/Crusher/" + run_image) as im:
        im.convert('RGB').save("image/Crusher/" + os.path.splitext(run_image)[0] + "_cr.jpg")
    os.unlink("image/Crusher/" + run_image)