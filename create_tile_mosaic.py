import printer_functions as pfns
from printer_functions import Dither_Method, Quality, Image_Scale

# Default is Floyd Steinberg - NOT RECOMMENDED FOR LARGE OUTPUTS, > (256, 256)
# dither_method = [Dither_Method.ERR_DIFF_FS, 
                  # Dither_Method.BLUE_NOISE, 
                  # Dither_Method.ERR_DIFF_PER, 
                  # Dither_Method.POLY_4, 
                  # Dither_Method.POLY_6, 
                  # Dither_Method.POLY_8, 
                  # Dither_Method.POLY_12, 
                  # Dither_Method.NEAREST]

# quality_icons are (32,32) which is rather large for a tile image, so default is NORMAL i.e. no quality icon
# quality_icon = [Quality.NORMAL, 
                  # Quality.LEGENDARY, 
                  # Quality.EPIC, 
                  # Quality.RARE, 
                  # Quality.UNCOMMON, 
                  # Quality.ALL]

# pre-crop the image, coords are left, upper, right, lower edges, from (0,0) coordinates starting at upper-left pixel
# image_crop = (x0, y0, x1, y1)

# image_final_size = (x_dim, y_dim) # Default final size is original image size. Recommended <(300,300) overall

# If image_final_size is default, FIT and CENTER will be equivalent
# image_scale: [Image_Scale.FIT, # scales image to fit within image_final_size
               # Image_Scale.CENTER, # crops the image at original size to fit within image_final_size
               # Image_Scale.X2_CENTER] # X2_CENTER doubles size then crops

# image_color = 1.0 # adjusts saturation - 1 is unchanged, > 1 is more saturated, 0 removes all (grayscale)

# image_contrast = 1.0 # adjusts contrast, 1 is unchanged, > 1 increases contrast

# image_brightness = 1.0 # adjusts brightness, 1 is unchanged, > 1 is brighter

# image_sharpness = 1.0 # adjusts sharpness, 1 is unchanged, > 1 is sharper image ;)

# background_color = (0,0,0) # Color seen behind any existing semi-transparency in the image, default Black #000000

# print_transparent_tiles = False # Whether to create tiles in areas of full (alpha < 5) transparency. 
# True will always result in rectangular mosaics 
# False will be visible areas of image only if transparency exists

pfns.create_factorio_tile_image(image_final_size=(128,128), image_scale = Image_Scale.FIT, background_color = (25,25,25), quality_icon=Quality.ALL)

# pfns.create_factorio_tile_image(quality_icon=Quality.NORMAL,
#                                  dither_method=Dither_Method.ERR_DIFF_FS,
#                                     # image_final_size=(360,250),
#                                      background_color = (192,192,192),
#                                      print_transparent_tiles = True,
#                                  #   image_color=1.5,
#                                    image_contrast=1.0)
