import os
import zlib
import base64
import json
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pandas as pd
import luadata
from enum import Enum, auto

class Dither_Method(Enum):
    BLUE_NOISE = auto()
    ERR_DIFF_PER = auto()
    ERR_DIFF_FS = auto()
    POLY_4 = auto()
    POLY_6 = auto()
    POLY_8 = auto()
    POLY_12 = auto()
    NEAREST = auto()

class Color_Metric(Enum):
    EUCLIDEAN = auto()
    NON_EUCLIDEAN = auto()

class Quality(Enum):
    LEGENDARY = auto()
    EPIC = auto()
    RARE = auto()
    UNCOMMON = auto()
    NORMAL = auto()
    ALL = auto()

class Image_Scale(Enum):
    FIT = auto()
    CENTER = auto()
    X2_CENTER = auto()
    INT_SCALE = auto()


# vertices of a tetrahedron about (0,0,0), each distance 1 from origin
unit_tetra_df = pd.DataFrame(data = {'ur': [0, 0, (8/9) ** (1/2), -(2/9) ** (1/2), -(2/9) ** (1/2)],
                                     'ug': [0, 0,              0,  (2/3) ** (1/2), -(2/3) ** (1/2)],
                                     'ub': [0, 1,           -1/3,            -1/3,            -1/3],
                                     'z_ind': [0, 1, 2, 3, 4]})

# golden ratio
phi = 2 / (1 + (5 ** (1/2)))

# vertices of an icosohedron (12 vertices) about (0,0,0), each distance 1 from origin
unit_dodeca_df = pd.DataFrame(data = {'ur': [0,      0,    0,    0,    0, -phi,  phi, -phi,  phi,   -1,   -1,    1,    1],
                                        'ug': [0,   -1,   -1,    1,    1,    0,    0,    0,    0, -phi,  phi, -phi,  phi],
                                        'ub': [0, -phi,  phi, -phi,  phi,   -1,   -1,    1,    1,    0,    0,    0,    0],
                                        'z_ind': [12, 0,   1,    6,    7,    2,    3,    8,    9,    4,    5,    10,  11]})

# vertices of an octahedron (6 vertices) about (0,0,0), each distance 1 from origin
unit_cubic_df = pd.DataFrame(data = {'ur': [0, -1,  1,  0,  0,  0,  0],
                                     'ug': [0,  0,  0,  0,  0, -1,  1],
                                     'ub': [0,  0,  0, -1,  1,  0,  0],
                                     'z_ind': [0, 1, 3, 5, 2, 4, 6]})

# root(1/3)
r_third = 3 ** (-1/2)

# vertices of a cube (8 vertices) about (0,0,0), each distance 1 from origin
unit_octa_df = pd.DataFrame(data = {'ur': [0, -r_third,  r_third, -r_third,  r_third, -r_third,  r_third, -r_third,  r_third],
                                    'ug': [0, -r_third,  r_third, -r_third,  r_third,  r_third, -r_third,  r_third, -r_third],
                                    'ub': [0, -r_third,  r_third,  r_third, -r_third,  r_third, -r_third, -r_third,  r_third],
                                    'z_ind': [0, 1, 2, 3, 4, 5, 6, 7, 8]})


# Creates a dictionary of nearest colors to color group centers, which look like (16 + i*32, 16 + j*32, 16 + k*32)
# These will be used as the sub-palette of colors to check when running nearest_palette_clr
def build_pal_dict(palette):
    global Euc_Dict
    global NE_Dict
    Euc_Dict = {}
    NE_Dict = {}
    for r in range(16, 255, 32):
        for g in range(16, 255, 32):
            for b in range(16, 255, 32):
                pal_tmp = palette.copy()
                # Find the distance of palette colors, and filter palette to those under a certain distance (relying on triangle inequality)
                dr2 = (pal_tmp['r.p'] - r) ** 2
                dg2 = (pal_tmp['g.p'] - g) ** 2
                db2 = (pal_tmp['b.p'] - b) ** 2
                r_bar = (pal_tmp['r.p'] + r) / 2
                pal_tmp_Euc = pal_tmp.copy()
                pal_tmp_NE = pal_tmp.copy()
                pal_tmp_Euc['dist'] = (dr2 + dg2 + db2) ** (1/2)
                pal_tmp_Euc = pal_tmp_Euc.sort_values(by = ['dist'])
                pal_tmp_Euc['dist'] -= min(pal_tmp_Euc.dist)
                # By the euclidean metric, a point (+16, +16, +16) away is ~27.7, so any colors that are closest to a pixel color must be
                # within 28 units of the closest color to the group colors in the center
                # filter down to this smaller palette before calculating collective closest colors
                pal_tmp_Euc = pal_tmp_Euc.loc[pal_tmp_Euc['dist'] < 30]
                Euc_Dict[(r, g, b)] = list(pal_tmp_Euc.index)
                pal_tmp_NE['dist'] = ((2 + r_bar/(256.0)) * dr2 + 4 * dg2 + (2 + (256 - r_bar)/(256.0)) * db2) ** (1/2)
                pal_tmp_NE = pal_tmp_NE.sort_values(by = ['dist'])
                pal_tmp_NE['dist'] -= min(pal_tmp_NE.dist)
                pal_tmp_NE = pal_tmp_NE.loc[pal_tmp_NE['dist'] < 60]
                NE_Dict[(r, g, b)] = list(pal_tmp_NE.index)
                #print(r, g, b)

clr_hex = {
    0: "00",
    64: "40",
    128: "80",
    192: "C0",
    255: "FF"
}

def pal_initialize():
    global pal_ideal
    pal_ideal = []

    i = 0
    for r in clr_hex:
        for g in clr_hex:
            for b in clr_hex:
                pal_ideal.append(pd.DataFrame(data = {'r.p':r, 'g.p':g, 'b.p':b, 'item': "tech-tile-" + clr_hex[r] + clr_hex[g]+ clr_hex[b]}, index = [i]))
                i += 1
    pal_ideal = pd.concat(pal_ideal)
    pal_ideal['index'] = list(range(0,125))

def nearest_palette_clr(pxls_df, palette):
    if  'pal_ideal' not in globals():
        pal_initialize()
    
    pxls_df[['0_R', '1_G', '2_B']] = pxls_df[['0_R', '1_G', '2_B']].clip(lower = 0, upper = 255)

    # For palette colors of the form (r,g,b) where r, g, b are in [0, 64, 128, 192, 255]
    # Euclidean and Non-Euclidean metrics produce same result for nearest color other
    # than the case where 0_R is 32, 96, and 160. 1_G and 2_B are always independent
    # since every pixel color lives in a cube of 8 palette colors
    # if metric == Color_Metric.EUCLIDEAN:
    #     pxls_df['r.p'] = ((pxls_df['0_R'].astype('uint') + 32) // 64 * 64).clip(upper = 255)
    # else:
    pxls_df['r.p'] = ((pxls_df['0_R'].astype('uint') + 31) // 64 * 64).clip(upper = 255)
    pxls_df['g.p'] = ((pxls_df['1_G'].astype('uint') + 31) // 64 * 64).clip(upper = 255)
    pxls_df['b.p'] = ((pxls_df['2_B'].astype('uint') + 31) // 64 * 64).clip(upper = 255)
    
    pxls_df = pxls_df.merge(pal_ideal, how = 'left')

    pxls_df[['x_ind', 'y_ind', '0_R', '1_G', '2_B', 'index', 'item', 'r.p', 'g.p', 'b.p']].sort_values(by = ['x_ind', 'y_ind'])
    
    return(pxls_df)
    

def nearest_palette_clr_old(pxls_df, palette, metric = Color_Metric.EUCLIDEAN):
    # Initializes Euc_Dict and NE_Dict if they don't already exist
    if 'Euc_Dict' not in globals():
        build_pal_dict(palette)
    
    pxls_df[['0_R', '1_G', '2_B']] = pxls_df[['0_R', '1_G', '2_B']].clip(lower = 0, upper = 255)

    # group pixel colors by (i,j,k) where (i,j,k) is center of 32^3 cube
    pxls_df['0_Re'] = (pxls_df['0_R'] // 32) * 32 + 16
    pxls_df['1_Ge'] = (pxls_df['1_G'] // 32) * 32 + 16
    pxls_df['2_Be'] = (pxls_df['2_B'] // 32) * 32 + 16

    df_grp = pxls_df.groupby(['0_Re', '1_Ge', '2_Be'])

    out_list = []

    for grp in df_grp.groups:
        r_c = grp[0]
        g_c = grp[1]
        b_c = grp[2]
        pal_tmp = palette.copy()
        if (metric == Color_Metric.EUCLIDEAN):
            pal_tmp = pal_tmp.merge(pd.DataFrame(data = {'index': Euc_Dict[(r_c, g_c, b_c)]}), how = 'inner')
        if (metric == Color_Metric.NON_EUCLIDEAN):
            pal_tmp = pal_tmp.merge(pd.DataFrame(data = {'index': NE_Dict[(r_c, g_c, b_c)]}), how = 'inner')
        df_sub = df_grp.get_group(grp)[['x_ind', 'y_ind', '0_R', '1_G', '2_B']].merge(pal_tmp, how = 'cross')
        # If there's only 1 possible color for these pixels, we don't need to find the closest palette color to each pixel and we can skip this part
        if len(pal_tmp) > 1:
            dr2 = (df_sub['r.p'] - df_sub['0_R']) ** 2
            dg2 = (df_sub['g.p'] - df_sub['1_G']) ** 2
            db2 = (df_sub['b.p'] - df_sub['2_B']) ** 2
            if (metric == Color_Metric.EUCLIDEAN):
                df_sub['dist'] = (dr2 + dg2 + db2) ** (1/2)
            elif (metric == Color_Metric.NON_EUCLIDEAN):
                r_bar = (df_sub['r.p'] + df_sub['0_R'])/2
                df_sub['dist'] = ((2 + r_bar/(256.0)) * dr2 + 4 * dg2 + (2 + (256 - r_bar)/(256.0)) * db2) ** (1/2)
            df_sub = df_sub.sort_values(by = ['dist']).groupby(['x_ind', 'y_ind']).nth(0)[['x_ind', 'y_ind', '0_R', '1_G', '2_B', 'index', 'item', 'r.p', 'g.p', 'b.p']]
        out_list.append(df_sub)

    df_final = pd.concat(out_list).sort_values(by = ['x_ind', 'y_ind'])
    print(df_final)
    return(df_final)


# Takes a PIL.Image and converts it to a form x_ind, y_ind, 0_R, 1_G, 2_B
# x_ind and y_ind will be centered around (1500, 360), and image will be contained within (3000, 720)
def image_to_df(im_orig):
    im = im_orig.copy()
    x_dim = im.size[0]
    y_dim = im.size[1]
    x_min = 1500 - x_dim // 2
    y_min = 360 - y_dim // 2

    im_arr = np.array(im)

    # reshapes the array into a 1-d array, which starts at 
    # upper left pixel, moving right, going R, G, B, R, G, B
    # so im_arr[0:3] is pixel (0,0), im_arr[3:6] is pixel (1,0), etc. x and y index from upper left
    im_arr = im_arr.reshape((x_dim * y_dim * 3))

    # list of x values from x_min to x_min + x_dim, each
    #  repeated three times because of the three colors
    x_coords = list(range(x_min, x_min + x_dim)) * 3
    x_coords.sort()
    # this list stacked y_dim times
    x_coords = x_coords * y_dim

    # y_coords are strictly increasing
    y_coords = list(range(y_min, y_min + y_dim)) * 3 * x_dim
    y_coords.sort()

    clr_coords = ["0_R", "1_G", "2_B"] * x_dim * y_dim

    # builds a dataframe with these three coordinate columns
    coords_df = pd.DataFrame(data = {'x_ind': x_coords, 'y_ind': y_coords, "clr_ind": clr_coords})

    # converts the 1-d image array into a dataframe 
    im_df = pd.DataFrame(im_arr, columns=["val"])

    # combines these two data frames to get indices + value
    im_df = pd.concat([coords_df, im_df], axis = 1)

    # dataframe with x_ind and y_ind as indices, 
    # three additional columns 0_R, 1_G, 2_B with integers 0 - 255
    im_df = im_df.pivot(index = ['x_ind','y_ind'],
                        columns = 'clr_ind',
                        values = 'val')

    im_df = im_df.reset_index(level = ['x_ind','y_ind'])[['x_ind', 'y_ind', '0_R', '1_G', '2_B']]

    return(im_df)

# exports an image with palette colors to a jpg, doubling the height of pixels
def export_im_df(im_df, file):
    x_dim = max(im_df.x_ind) - min(im_df.x_ind) + 1
    y_dim = max(im_df.y_ind) - min(im_df.y_ind) + 1
    im_df3 = pd.melt(frame = im_df.set_index(['x_ind','y_ind']).rename(columns = {'r.p': '0.R', 'g.p': '1.G', 'b.p': '2.B'}), 
                 id_vars = None, 
                 value_vars = ['0.R', '1.G', '2.B'], 
                 ignore_index = False).sort_values(by = ['y_ind', 'x_ind', 'variable']).astype({"value": 'uint8'})
    im_arr_out = np.array(im_df3['value']).reshape((y_dim, x_dim, 3))
    Image.fromarray(im_arr_out, mode = "RGB").resize((x_dim, y_dim * 2)).save(file)

# Load image from run folder, prepare image for processing with pre-cropping, sizing, quality icon
def prepare_image(
        # pre-crop the image, coords are left, upper, right, lower edges, from (0,0) coordinates starting at upper-left pixel
        image_crop: tuple[int, int, int, int] | None = None,
        # fit scales image to fit within frame, center crops the image at original size, x2 doubles size then crops
        image_scale: list[Image_Scale] = [Image_Scale.FIT, Image_Scale.CENTER, Image_Scale.X2_CENTER, Image_Scale.INT_SCALE],
        # four pass-through methods from ImageEnhance
        image_color: float = 1.0,
        image_contrast: float = 1.0,
        image_brightness: float = 1.0,
        image_sharpness: float = 1.0,
        # adds a quality icon to the print, default is legendary
        quality_icon: list[Quality] = [Quality.LEGENDARY, Quality.EPIC, Quality.RARE, Quality.UNCOMMON, Quality.NORMAL, Quality.ALL],
        image_final_size: tuple[int, int] | None = None,
        pixels_combine_vertical: int = 2,
        background_color: tuple[int] = (0,0,0),
        empty_transparency: bool = True
):
    run_image = os.listdir('image/Run/')[0]
    print(run_image)
    global im
    
    # im is Image object from Pillow, with transparency channel dropped if it exists
    im = Image.open("image/Run/" + run_image).copy()
    if im.mode == 'P':
        im = im.convert('RGBA')
    #.convert("RGBA")
    global orig_size
    orig_size = im.size
    # pastes image with transparency over black background; sometimes transparent images have strange pixels in transparent areas
    if im.mode == 'RGBA':
        im2 = Image.new(mode = 'RGBA', size = orig_size, color = background_color)
        im2.paste(im, mask = im)
        if empty_transparency:
            im2.putalpha(im.split()[3])
        im = im2
    
    if im.mode == 'RGB':
        im = im.convert('RGBA')
        im.putalpha(255)

    # im = im.convert('RGB')

    #print(image_scale)
    
    if not image_crop == None:
        im = im.crop(image_crop)
        orig_size = im.size

    if image_final_size == None:
        image_final_size = ((im.size[0] // 2) * 2, (im.size[1] // 2) * 2)

    if type(image_scale) is list:
        image_scale = image_scale[0]
    if image_scale == Image_Scale.FIT:
        im = ImageOps.contain(im, image_final_size)
    elif image_scale == Image_Scale.X2_CENTER:
        im = im.resize((orig_size[0] * 2, orig_size[1] * 2))
    
    # im.size will be (width, height)

    # removes last row/column of pixels to make resulting image 
    # have even number of rows, columns, for centering and /2 operations
    im = im.crop((0, 0, (im.size[0] // 2) * 2, (im.size[1] // 2) * 2))

    x_dim = im.size[0]
    y_dim = im.size[1]

    x_dim_final = image_final_size[0]
    y_dim_final = image_final_size[1]
    # Final image has resolution (3000, 1440), this finds the 
    # x and y pixels where the image should start to be centered
    x_min = (x_dim_final // 2) - x_dim // 2
    y_min = (y_dim_final // 2) - y_dim // 2

    print((x_dim, y_dim))
    if x_dim > x_dim_final:
        im = im.crop((-x_min, 0, x_dim + x_min, y_dim))
    
    x_dim = im.size[0]
    x_min = (x_dim_final // 2) - x_dim // 2
    
    if y_dim > y_dim_final:
        im = im.crop((0, -y_min, x_dim, y_dim + y_min))

    if pixels_combine_vertical == 2:
        y_dim = im.size[1] // 2
        y_min = (y_dim_final // 4) - y_dim // 2
    elif pixels_combine_vertical == 1:
        y_dim = im.size[1]
        y_min = (y_dim_final // 2) - y_dim // 2

    if type(quality_icon) is list:
        quality_icon = quality_icon[0]
    # print((x_min, y_min))
    if quality_icon != Quality.NORMAL:
        # im = im.convert('RGBA')
        match quality_icon:

            case Quality.LEGENDARY: 
                im_quality = Image.open("dat/Quality_legendary.png").copy()
            case Quality.EPIC:
                im_quality = Image.open("dat/Quality_epic.png").copy()
            case Quality.RARE:
                im_quality = Image.open("dat/Quality_rare.png").copy()
            case Quality.UNCOMMON:
                im_quality = Image.open("dat/Quality_uncommon.png").copy()    
            case Quality.ALL:
                im_quality = Image.open("dat/Any_quality.png").copy()
        im_quality = im_quality.convert('RGBA')
        im.paste(im_quality.resize((32, 32)), (3, im.size[1] - 35), im_quality.resize((32, 32)))
        # im = im.convert('RGB')

    if image_color != 1.0:
        im = ImageEnhance.Color(im).enhance(image_color)
    if image_contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(image_contrast)
    if image_brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(image_brightness)
    if image_sharpness != 1.0:
        im = ImageEnhance.Sharpness(im).enhance(image_sharpness)
    
    im_copy = im.copy()
    im_copy.putalpha(255)

    im_copy.show()
    im = im.resize((x_dim, y_dim))
    return(im)

def dither_blue_noise(im_df, color_scale = 30):

    blue_noise_im = Image.open("dat/HDR_RGB_0.png").convert("RGB")
    bn_df = image_to_df(blue_noise_im)
    bn_df['i'] = bn_df['x_ind'] % 256
    bn_df['j'] = bn_df['y_ind'] % 256
    bn_df = bn_df[['i', 'j', '0_R', '1_G', '2_B']]

    scale_fact = color_scale * 80 / 30
    bn_df['ur'] = scale_fact * (bn_df['0_R'].astype("int") - 128) / 256
    bn_df['ug'] = scale_fact * (bn_df['1_G'].astype("int") - 128) / 256
    bn_df['ub'] = scale_fact * (bn_df['2_B'].astype("int") - 128) / 256

    bn_df['ur'] = bn_df['ur'].astype("int")
    bn_df['ug'] = bn_df['ug'].astype("int")
    bn_df['ub'] = bn_df['ub'].astype("int")
    bn_df = bn_df[['i', 'j', 'ur', 'ug', 'ub']]

    im_df['i'] = (im_df['x_ind'] + 7) % 256
    im_df['j'] = (im_df['y_ind'] + 17) % 256

    im_df = im_df.merge(bn_df, how = 'left')
    im_df['0_R'] = (im_df['0_R'].astype("int") + im_df['ur']).astype("int")
    im_df['1_G'] = (im_df['1_G'].astype("int") + im_df['ug']).astype("int")
    im_df['2_B'] = (im_df['2_B'].astype("int") + im_df['ub']).astype("int")
    return(nearest_palette_clr(im_df, None))

def dither_POLY_4(im_df, color_scale = 30):
    # color_scale = 30
    im_df['z_ind'] = (im_df['x_ind'] + 2 * im_df['y_ind']) % 5
    im_df = im_df.merge(unit_tetra_df, how = 'left')
    im_df['0_R'] = (im_df['0_R'] + color_scale * im_df['ur']).astype("int")
    im_df['1_G'] = (im_df['1_G'] + color_scale * im_df['ug']).astype("int")
    im_df['2_B'] = (im_df['2_B'] + color_scale * im_df['ub']).astype("int")
    return(nearest_palette_clr(im_df, None))

def dither_POLY_12(im_df, color_scale = 30):
    # color_scale = 30
    im_df['z_ind'] = (im_df['x_ind'] + 5 * im_df['y_ind']) % 13
    
    im_df = im_df.merge(unit_dodeca_df, how = 'left')
    im_df['0_R'] = (im_df['0_R'] + color_scale * im_df['ur']).astype("int")
    im_df['1_G'] = (im_df['1_G'] + color_scale * im_df['ug']).astype("int")
    im_df['2_B'] = (im_df['2_B'] + color_scale * im_df['ub']).astype("int")
    return(nearest_palette_clr(im_df, None))

def dither_POLY_6(im_df, color_scale = 30):
    im_df['z_ind'] = (im_df['x_ind'] + 3 * im_df['y_ind']) % 7
    im_df = im_df.merge(unit_cubic_df, how = 'left')
    im_df['0_R'] = (im_df['0_R'] + color_scale * im_df['ur']).astype("int")
    im_df['1_G'] = (im_df['1_G'] + color_scale * im_df['ug']).astype("int")
    im_df['2_B'] = (im_df['2_B'] + color_scale * im_df['ub']).astype("int")
    return(nearest_palette_clr(im_df, None))

def dither_POLY_8(im_df, color_scale = 30):
    im_df['z_ind'] = (im_df['x_ind'] + 4 * im_df['y_ind']) % 9
    im_df = im_df.merge(unit_octa_df, how = 'left')
    im_df['0_R'] = (im_df['0_R'] + color_scale * im_df['ur']).astype("int")
    im_df['1_G'] = (im_df['1_G'] + color_scale * im_df['ug']).astype("int")
    im_df['2_B'] = (im_df['2_B'] + color_scale * im_df['ub']).astype("int")
    return(nearest_palette_clr(im_df, None))

def dither_ED_P16(im_df, color_scale = 30):
    out_set = []
    im_df_copy = im_df
    im_df['b_ind'] = (11 * im_df['x_ind'] + 7 * im_df['y_ind']) % 16
    error_weights = pd.DataFrame(data = {'dx': [1,    -1,  0,   1,  -1,  0,  1,  -1,  -2,   2,   0,   0],
                                         'dy': [1,     0,  1,  -1,   1, -1,  0,  -1,   0,   0,  -2,   2],
                                         'de': [1/12,1/6,1/6,1/12,1/12,1/6,1/6,1/12,1/18,1/18,1/18,1/18]})
    for b in range(16):
        print("Processing periodic error diffusion group",b+1,"out of 16.")
        im_out = im_df_copy.copy()
        im_out = im_out.loc[im_out['b_ind'] == b]
        im_out = nearest_palette_clr(im_out, None)
        out_set.append(im_out)
        im_err_sub = im_out.copy()
        im_err_sub = im_err_sub.merge(error_weights, how = 'cross')
        im_err_sub['er_r'] = (im_err_sub['0_R'].astype("int") - im_err_sub['r.p'].astype("int")) * im_err_sub['de']
        im_err_sub['er_g'] = (im_err_sub['1_G'].astype("int") - im_err_sub['g.p'].astype("int")) * im_err_sub['de']
        im_err_sub['er_b'] = (im_err_sub['2_B'].astype("int") - im_err_sub['b.p'].astype("int")) * im_err_sub['de']
        im_err_sub['x_ind'] = im_err_sub['x_ind'] + im_err_sub['dx']
        im_err_sub['y_ind'] = im_err_sub['y_ind'] + im_err_sub['dy']
        im_err_sub = im_err_sub[['x_ind', 'y_ind', 'er_r', 'er_g', 'er_b']].groupby(['x_ind', 'y_ind']).agg('sum')
        im_df_copy = im_df_copy.loc[im_df_copy['b_ind'] > b].merge(im_err_sub.reset_index(level = ['x_ind','y_ind']), how = 'left')
        
        im_df_copy['0_R'] = im_df_copy['0_R'].astype("int") + im_df_copy['er_r'].fillna(0).astype("int")
        im_df_copy['1_G'] = im_df_copy['1_G'].astype("int") + im_df_copy['er_g'].fillna(0).astype("int")
        im_df_copy['2_B'] = im_df_copy['2_B'].astype("int") + im_df_copy['er_b'].fillna(0).astype("int")
        im_df_copy = im_df_copy[['x_ind', 'y_ind', '0_R', '1_G', '2_B', 'b_ind']]
    df_out_ED = pd.concat(out_set)
    df_out_ED.sort_values(by = ['x_ind', 'y_ind']) 
    return(df_out_ED)   

def dither_ED_FS(im_df, color_scale = 30):
    x_min = min(im_df['x_ind'])
    x_max = max(im_df['x_ind'])
    y_min = min(im_df['y_ind'])
    y_max = max(im_df['y_ind'])

    out_set = []
    im_df_copy = im_df
    
    error_weights = pd.DataFrame(data = {'dx': [1,      0,    -1,    1],
                                         'dy': [0,      1,     1,    1],
                                         'de': [7/16, 5/16, 3/16, 1/16]})
    total_subs = 2*y_max + x_max - 2*y_min - x_min + 1
    print("Total subs to dither", total_subs)

    for k in range(2*y_min + x_min, 2*y_max + x_max + 1):
        k_shift = k - 2*y_min - x_min
        if (k_shift * 20) // total_subs > ((k_shift-1) * 20) // total_subs:
            print("Dithering sub", k_shift, "out of", total_subs)
        im_out = im_df_copy.copy()
        im_out = im_out.loc[2 * im_out['y_ind'] + im_out['x_ind'] == k]
        im_out = nearest_palette_clr(im_out, None)
        out_set.append(im_out)
        im_err_sub = im_out.copy()
        im_err_sub = im_err_sub.merge(error_weights, how = 'cross')
        im_err_sub['er_r'] = (im_err_sub['0_R'].astype("int") - im_err_sub['r.p'].astype("int")) * im_err_sub['de']
        im_err_sub['er_g'] = (im_err_sub['1_G'].astype("int") - im_err_sub['g.p'].astype("int")) * im_err_sub['de']
        im_err_sub['er_b'] = (im_err_sub['2_B'].astype("int") - im_err_sub['b.p'].astype("int")) * im_err_sub['de']
        im_err_sub['x_ind'] = im_err_sub['x_ind'] + im_err_sub['dx']
        im_err_sub['y_ind'] = im_err_sub['y_ind'] + im_err_sub['dy']
        im_err_sub = im_err_sub[['x_ind', 'y_ind', 'er_r', 'er_g', 'er_b']].groupby(['x_ind', 'y_ind']).agg('sum')
        im_df_copy = im_df_copy.loc[2 * im_df_copy['y_ind'] + im_df_copy['x_ind'] > k].merge(im_err_sub.reset_index(level = ['x_ind','y_ind']), how = 'left')
        im_df_copy['0_R'] = im_df_copy['0_R'].astype("int") + im_df_copy['er_r'].fillna(0).astype("int")
        im_df_copy['1_G'] = im_df_copy['1_G'].astype("int") + im_df_copy['er_g'].fillna(0).astype("int")
        im_df_copy['2_B'] = im_df_copy['2_B'].astype("int") + im_df_copy['er_b'].fillna(0).astype("int")
        im_df_copy = im_df_copy[['x_ind', 'y_ind', '0_R', '1_G', '2_B']]
    df_out_FS = pd.concat(out_set)
    df_out_FS.sort_values(by = ['x_ind', 'y_ind']) 
    return(df_out_FS)

def dither_Nearest(im_df, color_scale = 30):
    return(nearest_palette_clr(im_df, None))

dither_Dict = {
    Dither_Method.BLUE_NOISE: dither_blue_noise,
    Dither_Method.POLY_4: dither_POLY_4,
    Dither_Method.POLY_6: dither_POLY_6,
    Dither_Method.POLY_8: dither_POLY_8,
    Dither_Method.POLY_12: dither_POLY_12,
    Dither_Method.ERR_DIFF_PER: dither_ED_P16,
    Dither_Method.ERR_DIFF_FS: dither_ED_FS,
    Dither_Method.NEAREST: dither_Nearest
}

# Processes an image with various methods of matching pixels to palette colors, and exports each variation into image/Output
def create_im_samples(
        # Run using one or a list of dithering methods
        dither_method: list[Dither_Method] = [Dither_Method.BLUE_NOISE, Dither_Method.ERR_DIFF_PER, Dither_Method.POLY_4, Dither_Method.POLY_6, Dither_Method.POLY_8, Dither_Method.POLY_12, Dither_Method.NEAREST],
        # closeness metric is now deprecated
        # closeness_metric: list[Color_Metric] = [Color_Metric.EUCLIDEAN, Color_Metric.NON_EUCLIDEAN],
        # Adds a quality icon to the print, default is legendary. Only uses the first in the list
        quality_icon: list[Quality] = [Quality.LEGENDARY, Quality.EPIC, Quality.RARE, Quality.UNCOMMON, Quality.NORMAL, Quality.ALL],
        # Pre-crop the image, coords are left, upper, right, lower edges, from (0,0) coordinates starting at upper-left pixel
        image_crop: tuple[int, int, int, int] | None = None,
        # fit scales image to fit within frame, center crops the image at original size, x2 doubles size then crops. Only uses the first in the list
        image_scale: list[Image_Scale] = [Image_Scale.FIT, Image_Scale.CENTER, Image_Scale.X2_CENTER, Image_Scale.INT_SCALE],
        image_color: float = 1.0,
        image_contrast: float = 1.0,
        image_brightness: float = 1.0,
        image_sharpness: float = 1.0
):
    im = prepare_image(image_crop = image_crop,
                  image_scale = image_scale,
                  quality_icon = quality_icon,
                  image_color = image_color,
                  image_contrast = image_contrast,
                  image_brightness = image_brightness,
                  image_sharpness = image_sharpness,
                  image_final_size = (3000, 1440))
    
    im = im.convert('RGB')

    # palette = pd.read_csv("dat/Final_Palette.csv", index_col = False)    
    
    # if 'Euc_Dict' not in globals():
    #     build_pal_dict(palette)
    
    out_images = os.listdir('image/Output')
    out_images.remove('.gitignore')

    for out_image in out_images:
        os.unlink('image/Output/' + out_image)

    x_dim = im.size[0]
    y_dim = im.size[1]

    # Final image has resolution (3000, 1440), this finds the 
    # x and y pixels where the image should start to be centered
    x_min = 1500 - x_dim // 2
    y_min = 360 - y_dim // 2

    im.resize((x_dim, y_dim * 2)).save("image/Output/000_Adjusted_Original.jpg")

    if not type(dither_method) is list:
        dither_method = [dither_method]
    # if not type(closeness_metric) is list:
    #     closeness_metric = [closeness_metric]
    
    im_df = image_to_df(im)
    
    for method in dither_method:
        # for metric in closeness_metric:
        im_df_copy = im_df.copy()
        im_df_copy = dither_Dict[method](im_df_copy)
                
        match method:
            case Dither_Method.BLUE_NOISE:
                print("Exporting BLUE_NOISE Closest Color")
                export_im_df(im_df_copy, "image/Output/BLUE_NOISE.jpg")

            case Dither_Method.ERR_DIFF_PER:
                print("Exporting ERR_DIFF_PER Closest Color")

                export_im_df(im_df_copy, "image/Output/ERR_DIFF_PER.jpg")

            case Dither_Method.ERR_DIFF_FS:
                print("Exporting ERR_DIFF_FS Closest Color")

                export_im_df(im_df_copy, "image/Output/ERR_DIFF_FS.jpg")

            case Dither_Method.POLY_4:
                print("Exporting POLY_4 Dithered")
                export_im_df(im_df_copy, "image/Output/POLY_4.jpg")
        
            case Dither_Method.POLY_12:
                print("Exporting POLY_12 Dithered")
                export_im_df(im_df_copy, "image/Output/POLY_12.jpg")
        
            case Dither_Method.POLY_6:
                print("Exporting POLY_6 Dithered")
                export_im_df(im_df_copy, "image/Output/POLY_6.jpg")

            case Dither_Method.POLY_8:
                print("Exporting POLY_8 Dithered")
                export_im_df(im_df_copy, "image/Output/POLY_8.jpg")

            case Dither_Method.NEAREST:
                print("Exporting Nearest Color")
                export_im_df(im_df_copy, "image/Output/NEAREST.jpg")

def create_factorio_printer_image(
        # runs the arg or the first arg in the list
        dither_method: list[Dither_Method] = [Dither_Method.BLUE_NOISE, Dither_Method.ERR_DIFF_PER, Dither_Method.POLY_4, Dither_Method.POLY_6, Dither_Method.POLY_8, Dither_Method.POLY_12, Dither_Method.NEAREST],
        # closeness_metric is now deprecated
        # closeness_metric: list[Color_Metric] = [Color_Metric.EUCLIDEAN, Color_Metric.NON_EUCLIDEAN],
        # adds a quality icon to the print, default is legendary
        quality_icon: list[Quality] = [Quality.LEGENDARY, Quality.EPIC, Quality.RARE, Quality.UNCOMMON, Quality.NORMAL, Quality.ALL],
        # pre-crop the image, coords are left, upper, right, lower edges, from (0,0) coordinates starting at upper-left pixel
        image_crop: tuple[int, int, int, int] | None = None,
        # fit scales image to fit within frame, center crops the image at original size, x2 doubles size then crops
        image_scale: list[Image_Scale] = [Image_Scale.FIT, Image_Scale.CENTER, Image_Scale.X2_CENTER, Image_Scale.INT_SCALE],
        image_color: float = 1.0,
        image_contrast: float = 1.0,
        image_brightness: float = 1.0,
        image_sharpness: float = 1.0
):
    

    if type(dither_method) is list:
        dither_method = dither_method[0]
    # if type(closeness_metric) is list:
    #     closeness_metric = closeness_metric[0]
    if type(quality_icon) is list:
        quality_icon = quality_icon[0]
    if type(image_scale) is list:
        image_scale = image_scale[0]

    # loads image 'im' in globals
    im = prepare_image(image_crop = image_crop,
                  image_scale = image_scale,
                  quality_icon = quality_icon,
                  image_color = image_color,
                  image_contrast = image_contrast,
                  image_brightness = image_brightness,
                  image_sharpness = image_sharpness,
                  image_final_size = (3000, 1440))
    im = im.convert('RGB')

    # palette = pd.read_csv("dat/Final_Palette.csv", index_col = False)    

    # blueprint template for 1 combinator with complete signals
    with open('dat/bp_comb_template.txt', 'r') as orig_bp:
        bp_lines = orig_bp.read()

    bp_decoded = json.loads(zlib.decompress(
        base64.b64decode(bp_lines[1:])).decode('utf8'))

    bp_split = luadata.serialize(bp_decoded, encoding = 'utf-8', indent = "\t").split("\n")

    bp_entity_template = bp_split[11:-5]

    # if 'Euc_Dict' not in globals():
    #     build_pal_dict(palette)
    
    x_dim = im.size[0]
    y_dim = im.size[1]

    # Final image has resolution (3000, 1440), this finds the 
    # x and y pixels where the image should start to be centered
    x_min = 1500 - x_dim // 2
    y_min = 360 - y_dim // 2

    im_df = image_to_df(im)
    im_df = dither_Dict[dither_method](im_df)
            
    # im_df_copy = im_df.copy()
    match dither_method:
        case Dither_Method.BLUE_NOISE:
            dither_method_desc = 'Dithered with BLUE_NOISE'
                
        case Dither_Method.ERR_DIFF_PER:
            dither_method_desc = 'Colors matched with ERR_DIFF_PER'

        case Dither_Method.ERR_DIFF_FS:
            dither_method_desc = 'Colors matched with ERR_DIFF_FS'
            
        case Dither_Method.POLY_4:
            dither_method_desc = 'Dithered with POLY_4 dithering'

        case Dither_Method.POLY_12:
            dither_method_desc = 'Dithered with POLY_12 dithering'

        case Dither_Method.POLY_6:
            dither_method_desc = 'Dithered with POLY_6 dithering'

        case Dither_Method.POLY_8:
            dither_method_desc = 'Dithered with POLY_8 dithering'

        case Dither_Method.NEAREST:
            dither_method_desc = 'Closest palette color'
        
    # Will appear in the 'description' box of the blueprint
    bp_description = ['\t\tdescription = "Original Size: ' + str((orig_size[0], orig_size[1])) + '\\', 
                      'Final Size: ' + str((im.size[0], im.size[1] * 2)) + '\\',
                      dither_method_desc + '\\', 
                      'Color distance using Euclidean metric\\',
                      '\\',
                      'made for the Factorio Printer v3, by travvo",']
    
    existing_bps = pd.DataFrame(data = {'bp_names': os.listdir("Blueprint Out/")})
    bp_indices = list(existing_bps['bp_names'].str.replace("^BP_([0-9]{1,2})_.+$", "\\1", regex = True).astype('int'))
    bp_indices.sort()
    bp_index = bp_indices[-1] + 1
    

    bp_index1 = bp_index % 10
    bp_index10 = bp_index // 10

    bp_icons = ['\t\ticons = {', 
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              '\t\t\t\t\tname = "signal-I",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 1,', 
             '\t\t\t},',
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              '\t\t\t\t\tname = "signal-M",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 2,', 
             '\t\t\t},',
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              f'\t\t\t\t\tname = "signal-{bp_index10}",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 3,', 
             '\t\t\t},',
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              f'\t\t\t\t\tname = "signal-{bp_index1}",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 4,', 
             '\t\t\t},',
            '\t\t},']
    
    bp_out = ['{', '\tblueprint = {'] + bp_description + bp_icons + ['\t\tentities = {']

    df_full = pd.DataFrame(data = {'y_ind':range(720)}).merge(pd.DataFrame(data = {'signal_index':range(750)}), how = 'cross')
    df_fin = im_df.sort_values(['y_ind', 'x_ind'])
    df_fin['quality_index'] = 4 - df_fin['x_ind'] % 5
    df_fin['signal_index'] = 5 * (149 - df_fin['x_ind'] // 20) + df_fin['quality_index']
    df_fin['power_index'] = 3 - (df_fin['x_ind'] % 20) // 5
    df_fin['count'] = df_fin['index'] * (125 ** df_fin['power_index'])
    df_fin = df_fin.groupby(['y_ind', 'signal_index']).agg({'count': ['sum']}).reset_index(level = ['y_ind', 'signal_index']).droplevel(1, axis = 1)

    df_fin = df_fin.merge(df_full, how = 'outer')

    df_fin['count'] = df_fin['count'].fillna(0).astype("int64")

    df_fin_grps = df_fin.groupby('y_ind')

    for y in range(720):
        
        if y in range(0,720,36):
            print("Compiling blueprint lines, progress: " + str(100 * y // 720) + " %")

        comb_x_pos = 7.5 - (y % 8)
        comb_y_pos = (y // 8) + 0.5
        entity_lines = ['\t\t\t{', 
                        f'\t\t\t\tentity_number = {y + 1},',
                        '\t\t\t\tname = "constant-combinator",',
                        '\t\t\t\tposition = {',
                        f'\t\t\t\t\tx = {comb_x_pos},', 
                        f'\t\t\t\t\ty = {comb_y_pos},', 
                        '\t\t\t\t},']
        if y in [9, 14, 705, 710]:
            entity_lines = entity_lines + ['\t\t\t\tdirection = 4,']
        entity_lines = entity_lines + ['\t\t\t\tcontrol_behavior = {', 
                                        '\t\t\t\t\tsections = {', 
                                        '\t\t\t\t\t\tsections = {', 
                                        '\t\t\t\t\t\t\t{', 
                                        '\t\t\t\t\t\t\t\tindex = 1,']
        df_sub = df_fin_grps.get_group(y).copy()
        if sum(df_sub['count']) > 0:
            # 8 elements for each unique signal
            df_sub = df_sub.merge(pd.DataFrame(data = {'signal_row':range(8)}), how = 'cross')
            df_sub['count_str'] = '\t\t\t\t\t\t\t\t\t\tcount = ' + df_sub['count'].map(str) + ','
            # 6000 lines from the template file, 8 * 750
            df_sub['out_str'] = bp_entity_template[14:-6]
            # filter df_sub down to rows where count > 0. 
            # This means area out of the picture takes up almost 
            # no space in the blueprint but will print as black border
            df_sub = df_sub.loc[df_sub['count'] > 0]
            # Replace template count with newly constructed counts, only in the correct row for each signal
            df_sub['out_str'] = np.where(df_sub['signal_row'] == 6, df_sub['count_str'], df_sub['out_str'])
            entity_lines = entity_lines + ['\t\t\t\t\t\t\t\tfilters = {'] + df_sub['out_str'].to_list() + ['\t\t\t\t\t\t\t\t},']
            
        bp_out = bp_out + entity_lines + bp_entity_template[-5:]    

    bp_name = os.path.splitext(os.listdir("image/Run")[0])[0]

    bp_out = bp_out + ['\t\t},', 
                    '\t\titem = "blueprint",', 
                    f'\t\tlabel = "{bp_name}",', 
                    '\t\tversion = 562949956108288,', 
                    '\t},', 
                    '}']
    
    bp_exp_name = "BP_" + str(bp_index) + "_" + bp_name + "_" + dither_method.name + ".txt"
    
    sep = "\n"
    bp_out2 = sep.join(bp_out)

    bp_out3 = luadata.unserialize(bp_out2)
    bp_lines10 = json.dumps(bp_out3, separators=(',',':'))
    bp_lines11 = zlib.compress(bp_lines10.encode('utf-8'), level = 9)
    bp_lines12 = base64.b64encode(bp_lines11).decode('utf-8')
    bp_lines13 = '0' + bp_lines12

    file1 = open("Blueprint Out/" + bp_exp_name, "w")
    # file1.writelines(bp_lines13[:1000000])
    file1.writelines(bp_lines13)
    file1.close()

def create_factorio_tile_image(
        # runs the arg or the first arg in the list
        dither_method: list[Dither_Method] = [Dither_Method.ERR_DIFF_FS, Dither_Method.BLUE_NOISE, Dither_Method.ERR_DIFF_PER, Dither_Method.POLY_4, Dither_Method.POLY_6, Dither_Method.POLY_8, Dither_Method.POLY_12, Dither_Method.NEAREST],
        # closeness_metric is now deprecated
        # closeness_metric: list[Color_Metric] = [Color_Metric.EUCLIDEAN, Color_Metric.NON_EUCLIDEAN],
        # adds a quality icon to the print, default is legendary
        quality_icon: list[Quality] = [Quality.NORMAL, Quality.LEGENDARY, Quality.EPIC, Quality.RARE, Quality.UNCOMMON, Quality.ALL],
        # pre-crop the image, coords are left, upper, right, lower edges, from (0,0) coordinates starting at upper-left pixel
        image_crop: tuple[int, int, int, int] | None = None,
        # fit scales image to fit within frame, center crops the image at original size, x2 doubles size then crops
        image_scale: list[Image_Scale] = [Image_Scale.FIT, Image_Scale.CENTER, Image_Scale.X2_CENTER, Image_Scale.INT_SCALE],
        image_color: float = 1.0,
        image_contrast: float = 1.0,
        image_brightness: float = 1.0,
        image_sharpness: float = 1.0,
        image_final_size: tuple[int, int] | None = None,
        background_color: tuple[int] = (0,0,0),
        print_transparent_tiles: bool = False,
):
    
    if type(dither_method) is list:
        dither_method = dither_method[0]
    # if type(closeness_metric) is list:
    #     closeness_metric = closeness_metric[0]
    if type(quality_icon) is list:
        quality_icon = quality_icon[0]
    if type(image_scale) is list:
        image_scale = image_scale[0]



    # loads image 'im' in globals
    im = prepare_image(image_crop = image_crop,
                  image_scale = image_scale,
                  quality_icon = quality_icon,
                  image_color = image_color,
                  image_contrast = image_contrast,
                  image_brightness = image_brightness,
                  image_sharpness = image_sharpness,
                  image_final_size = image_final_size,
                  pixels_combine_vertical = 1,
                  background_color = background_color,
                  empty_transparency = not print_transparent_tiles)
    
    im_orig = im.copy()
    # r,g,b,a = im.split()
    # print(np.array(a))
    im = im.convert('RGB')
    im_df = image_to_df(im)
    im_df = dither_Dict[dither_method](im_df)
    im_df['x_ind'] = im_df['x_ind'] - min(im_df['x_ind'])
    im_df['y_ind'] = im_df['y_ind'] - min(im_df['y_ind'])
    # print(im_df)

    bp_description = ['\t\tdescription = "Original Size: ' + str((orig_size[0], orig_size[1])) + '\\', 
                      'Final Size: ' + str((im.size[0], im.size[1])) + '\\',
                      'Color distance using Euclidean metric\\',
                      '\\',
                      'made for the Technicolor Tiles v1.0.0, by travvo",']
    
    existing_bps = pd.DataFrame(data = {'bp_names': os.listdir("Blueprint Out/")})
    bp_indices = list(existing_bps['bp_names'].str.replace("^BP_([0-9]{1,2})_.+$", "\\1", regex = True).astype('int'))
    bp_indices.sort()
    bp_index = bp_indices[-1] + 1
    

    bp_index1 = bp_index % 10
    bp_index10 = bp_index // 10

    bp_icons = ['\t\ticons = {', 
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              '\t\t\t\t\tname = "signal-I",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 1,', 
             '\t\t\t},',
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              '\t\t\t\t\tname = "signal-M",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 2,', 
             '\t\t\t},',
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              f'\t\t\t\t\tname = "signal-{bp_index10}",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 3,', 
             '\t\t\t},',
             '\t\t\t{', 
              '\t\t\t\tsignal = {', 
              '\t\t\t\t\ttype = "virtual",', 
              f'\t\t\t\t\tname = "signal-{bp_index1}",', 
              '\t\t\t\t},', 
             '\t\t\t\tindex = 4,', 
             '\t\t\t},',
            '\t\t},']
    
    bp_out = ['{', '\tblueprint = {'] + bp_description + bp_icons + ['\t\ttiles = {']
    # im_check = Image.merge(mode = 'RGBA', bands = [r,g,b,a])
    for tile_row in range(len(im_df)):
        x = im_df['x_ind'].iloc[tile_row]
        y = im_df['y_ind'].iloc[tile_row]
        alpha = im_orig.getpixel((x,y))[3]
        tile = im_df['item'].iloc[tile_row]
        # print(x)
        if alpha > 4 or print_transparent_tiles:
            bp_out = bp_out + ['\t\t\t{', 
                                '\t\t\t\tposition = {',
                                f'\t\t\t\t\tx = {x},',
                                f'\t\t\t\t\ty = {y}',
                                '\t\t\t\t},',
                                f'\t\t\t\tname = "{tile}"',
                                '\t\t\t},']
    
    
    bp_name = os.path.splitext(os.listdir("image/Run")[0])[0]

    bp_out = bp_out + ['\t\t},', 
                    '\t\titem = "blueprint",', 
                    f'\t\tlabel = "{bp_name}",', 
                    '\t\tversion = 562949956108288,', 
                    '\t},', 
                    '}']
    
    bp_exp_name = "BP_" + str(bp_index) + "_" + bp_name + "_tile mosaic.txt"
    
    # for i in range(len(bp_out)):
    #     print(bp_out[i])
    
    sep = "\n"
    bp_out2 = sep.join(bp_out)

    bp_out3 = luadata.unserialize(bp_out2)
    bp_lines10 = json.dumps(bp_out3, separators=(',',':'))
    bp_lines11 = zlib.compress(bp_lines10.encode('utf-8'), level = 9)
    bp_lines12 = base64.b64encode(bp_lines11).decode('utf-8')
    bp_lines13 = '0' + bp_lines12

    file1 = open("Blueprint Out/" + bp_exp_name, "w")
    # file1.writelines(bp_lines13[:1000000])
    file1.writelines(bp_lines13)
    file1.close()
