import printer_functions as pfns
#pfns.prepare_image(image_color=1.5, image_contrast = 1.5)

dither = ["Blue Noise", "Error Diffusion", "Tetrahedral", "Cubic", "Octahedral", "Dodecahedral", "Nearest Color"]

# ['Euclidean', 'Non-Euclidean']
metric = ['Non-Euclidean']

# ['legendary', 'epic', 'rare', 'uncommon', 'normal', 'all']
qual = ['legendary']

# im_crop = (0, 0, 1000, 1000) left, upper, right, lower pixel coords to crop.
#    cropping happens before image scale

im_crop = None

# im_scale = ["fit" - scale image to fit within frame, as large as possible, 
#               "center" - keep original size, cropping if necessary to fit, 
#               "x2 center" - first double size, then crop if necessary to fit]
im_scale = ['fit']

# im_color: 0.0 is grayscale, 1.0 is normal color, > 1.0 increases saturation
im_color = 1.0

# im_contrast: 0.0 is flat gray, 1.0 is normal, > 1.0 increases contrast
im_contrast = 1.0

# im_brightness: 0.0 is black, 1.0 is normal, > 1.0 increases brightness
im_brightness = 1.0

# im_sharpness: 1.0 is normal, > 1.0 increases sharpness
im_sharpness = 1.0


pfns.create_im_samples(dither_method = dither, 
                       closeness_metric = metric, 
                       quality_icon = qual, 
                       image_crop = im_crop, 
                       image_scale = im_scale,
                       image_color = im_color,
                       image_contrast = im_contrast,
                       image_brightness = im_brightness,
                       image_sharpness = im_sharpness)

