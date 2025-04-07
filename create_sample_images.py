import printer_functions as pfns
#pfns.prepare_image(image_color=1.5, image_contrast = 1.5)

# dith = ["Blue Noise", "Error Diffusion", "Tetrahedral", "Cubic", "Octahedral", "Dodecahedral", "Nearest Color"]
dither = ["Tetrahedral", "Cubic", "Octahedral", "Dodecahedral"]

# ['Euclidean', 'Non-Euclidean']
metric = ['Non-Euclidean']

# ['legendary', 'epic', 'rare', 'uncommon', 'normal', 'all']
qual = ['legendary']

# im_crop = (0, 0, 1000, 1000) left, upper, right, lower pixel coords to crop.
#    cropping happens before image scale

im_crop = None
#im_crop = (0, 0, 935, 400)

# im_scale = ["fit" - scale image to fit within frame, as large as possible, 
#               "center" - keep original size, cropping if necessary to fit, 
#               "x2 center" - first double size, then crop if necessary to fit]
im_scale = ['fit']

pfns.create_im_samples(dither_method = dither, 
                       closeness_metric = metric, 
                       quality_icon = qual, 
                       image_crop = im_crop, 
                       image_scale = im_scale,
                       image_color = 2)

