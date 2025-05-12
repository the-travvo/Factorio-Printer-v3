import printer_functions as pfns
from printer_functions import Dither_Method, Quality, Image_Scale

#pfns.prepare_image(image_color=1.5, image_contrast = 1.5)

dither = [Dither_Method.BLUE_NOISE, 
          Dither_Method.ERR_DIFF_PER, 
          Dither_Method.ERR_DIFF_FS, 
          Dither_Method.POLY_4, 
          Dither_Method.POLY_6, 
          Dither_Method.POLY_8,
          Dither_Method.POLY_12, 
          Dither_Method.NEAREST]

# [Quality.LEGENDARY, Quality.EPIC, Quality.RARE, Quality.UNCOMMON, Quality.NORMAL, Quality.ALL]
qual = [Quality.ALL]

# im_crop = (0, 0, 1000, 1000) left, upper, right, lower pixel coords to crop.
#    cropping happens before image scale

im_crop = None
# im_scale = [Image_Scale.FIT - scale image to fit within frame, as large as possible, 
#               Image_Scale.CENTER - keep original size, cropping if necessary to fit, 
#               Image_Scale.X2_CENTER - first double size, then crop if necessary to fit,
#               Image_Scale.INT_SCALE - (not built) scale ]
im_scale = Image_Scale.CENTER

# im_color: 0.0 is grayscale, 1.0 is normal color, > 1.0 increases saturation
im_color = 1.0

# im_contrast: 0.0 is flat gray, 1.0 is normal, > 1.0 increases contrast
im_contrast = 1.0

# im_brightness: 0.0 is black, 1.0 is normal, > 1.0 increases brightness
im_brightness = 1.0

# im_sharpness: 1.0 is normal, > 1.0 increases sharpness
im_sharpness = 1.0


pfns.create_im_samples(dither_method = [Dither_Method.BLUE_NOISE], 
                       quality_icon = qual, 
                       image_crop = im_crop, 
                       image_scale = im_scale,
                       image_color = im_color,
                       image_contrast = im_contrast,
                       image_brightness = im_brightness,
                       image_sharpness = im_sharpness)

