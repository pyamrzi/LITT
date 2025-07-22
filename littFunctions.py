# This file will contain the functions used in the LITT project.
# These scripts were initially developed by Benjamin Chao in Dr. LaViolette's lab.
# They have been modified by Pouya Mirzaei.

import numpy as np
import skimage as ski
import histomicstk as htk
import tqdm
import pyvips
import multiprocessing as mp
import lavlab
from lavlab.omero.images import get_large_recon
from lavlab.omero.rois import get_image_shapes_as_points
from skimage.filters import threshold_multiotsu, butterworth
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv, rgb2hed
from skimage.draw import polygon as skimage_polygon
from valis.preprocessing import create_tissue_mask_with_jc_dist 
import math
from tqdm import tqdm

def create_filled_polygon_mask_skimage(vertices, height, width, mask_value=True, dtype=bool):
    if vertices is None or len(vertices) == 0:
        print("Warning: Empty list of vertices provided.")
        return np.zeros((height, width), dtype=dtype)
    # Initialize the mask
    mask = np.zeros((height, width), dtype=dtype)
    vertex_array = np.array(vertices)
    rows = vertex_array[:, 0] # .clip(0, height - 1) # y-coordinates are rows
    cols = vertex_array[:, 1] # .clip(0, width - 1)  # x-coordinates are columnss.
    rr, cc = skimage_polygon(rows, cols, shape=(height, width))
    mask[rr, cc] = mask_value
    return mask

def get_combined_masks(image, obj_color, ds=10, name="", show_progress=True):
    # Extract ROIs and shape info only once
    shapeinfo = get_image_shapes_as_points(image, point_downsample_factor=1, scale_factor=ds)
    yx_shape = image.getSizeY() / ds, image.getSizeX() / ds
    yx_shape_rounded = tuple(math.ceil(x) for x in yx_shape)
    indices = [i for i, (_, color, _) in enumerate(shapeinfo) if color == obj_color]
    range_func = tqdm if show_progress else lambda x, **kwargs: x
    def create_mask(i):
        poly_points = shapeinfo[i][2]
        mask = create_filled_polygon_mask_skimage(poly_points, *yx_shape_rounded)
        return mask.astype(bool)
    combmasks = [create_mask(i) for i in range_func(indices, desc=f"Extracting {name} ROIs")]
    return combmasks

def get_combined_shape_points(image, obj_color, ds=10, name="", show_progress=True):
    shapeinfo = get_image_shapes_as_points(image, point_downsample_factor=1, scale_factor=ds)
    indices = [i for i, (_, color, _) in enumerate(shapeinfo) if color == obj_color]
    range_func = tqdm if show_progress else lambda x, **kwargs: x

    # Get image size (downscaled)
    height = math.ceil(image.getSizeY() / ds)
    width = math.ceil(image.getSizeX() / ds)

    def transform_point(y, x, height, width):
        # Rotate 90deg CW
        y_rot, x_rot = x, height - 1 - y
        # Flip horizontally
        x_flip = width - 1 - x_rot
        return [y_rot, x_flip]

    transformed_shape_points = []
    for i in range_func(indices, desc=f"Extracting {name} ROIs"):
        poly_points = shapeinfo[i][2]
        transformed_poly = [
            transform_point(y, x, height, width)
            for y, x in poly_points
        ]
        transformed_shape_points.append(
            # Save as np.float32 array
            np.array(transformed_poly, dtype=np.float32)
        )
    return transformed_shape_points

#def get_combined_shape_points(image, obj_color, ds=10, name="", show_progress=True):
#    import math
#    from tqdm import tqdm
#    # Extract ROIs and shape info only once
#    shapeinfo = get_image_shapes_as_points(image, point_downsample_factor=1, scale_factor=ds)
#    # Find indices for matching color
#    indices = [i for i, (_, color, _) in enumerate(shapeinfo) if color == obj_color]
#    range_func = tqdm if show_progress else lambda x, **kwargs: x
#    # Instead of mask, return polygon points
#    shape_points = [shapeinfo[i][2] for i in range_func(indices, desc=f"Extracting {name} ROIs")]
#    return shape_points

def get_lr_in_vips(image_obj, ds=10):
    # get lr and convert it to vips_lr
    lr = get_large_recon(image_obj, ds=ds)
    lr_array = np.ascontiguousarray(lr, dtype=np.uint8)
    height, width, bands = lr_array.shape
    vips_lr = pyvips.Image.new_from_memory(
        lr_array.tobytes(),
        width,
        height,
        bands,
        'uchar'  # or the appropriate format for your dtype
    )
    return vips_lr

def filter_shapes_by_overlap(difbin, difbin_bed):
    # Label connected components in difbin
    labeled = label(difbin)
    result = np.zeros_like(difbin, dtype=bool)
    for region in regionprops(labeled):
        # Get coordinates of the region
        coords = region.coords
        # Count how many pixels in that region are also 1 in difbin_bed
        overlap = np.sum(difbin_bed[tuple(coords.T)])
        if overlap > (len(coords) / 2):
            # Keep the region
            result[tuple(coords.T)] = True
    return result

def get_LR_tissue_mask(image):
    try:
        image_down = image.resize(.1)
    except TypeError:
        print("make sure youre using a pyvips image...")
    
    mask, _ = create_tissue_mask_with_jc_dist(image_down.numpy())
    maskpv = pyvips.Image.new_from_array(mask)
    scale_x = image.width / mask.shape[1]
    scale_y = image.height / mask.shape[0]
    mask_big = maskpv.resize(scale_x, vscale = scale_y)
    return mask_big, mask

def get_tile_list(large_recon, down_factor=100):
    tiles = []
    width = large_recon.shape[0]
    height = large_recon.shape[1]
    edge_len = int(np.rint(np.sqrt(down_factor)))
    start_x = 0
    x = 0
    while start_x < width:
        end_x = min(start_x + edge_len, width)
        start_y = 0
        y = 0
        while start_y < height:
            end_y = min(start_y + edge_len, height)
            tiles.append(((start_x, start_y, end_x, end_y),(x,y)))
            start_y = end_y
            y += 1
        start_x = end_x
        x += 1
    return tiles

def DABLR(large_recon_pv, mask=None, down_factor=10000):
    # Get tile list using only shape, not full array
    shape = large_recon_pv.shape if hasattr(large_recon_pv, 'shape') else np.array(large_recon_pv).shape
    tiles = get_tile_list(shape, down_factor)  # Adapt get_tile_list to take shape only
    if mask is None:
        # Generate mask on-demand per tile if possible
        lr_mask_full = get_LR_tissue_mask(large_recon_pv)[0]
        mask_is_array = isinstance(lr_mask_full, np.ndarray)
    else:
        lr_mask_full = mask
        mask_is_array = isinstance(mask, np.ndarray)
    # Get max tile coordinates for summary arrays
    ((_, _, _, _), (max_x, max_y)) = tiles[-1]
    cell = np.zeros((max_x + 1, max_y + 1), dtype=np.float32)
    dabh = np.zeros((max_x + 1, max_y + 1), dtype=np.float32)
    rat = np.zeros((max_x + 1, max_y + 1), dtype=np.float32)
    hemh = np.zeros((max_x + 1, max_y + 1), dtype=np.float32)
    mask_arr = np.zeros((max_x + 1, max_y + 1), dtype=np.float32)
    for ((start_x, start_y, end_x, end_y), (x, y)) in tqdm.tqdm(tiles):
        # Only load/process needed tile
        tile = large_recon_pv[start_x:end_x, start_y:end_y, :]
        if hasattr(tile, 'numpy'):
            tile = tile.numpy()
        dab_tile = None
        hem_tile = None
        # Mask tile
        if mask_is_array:
            mask_tile = lr_mask_full[start_x:end_x, start_y:end_y]
        else:
            mask_tile = get_LR_tissue_mask(tile)[0]
            if hasattr(mask_tile, 'numpy'):
                mask_tile = mask_tile.numpy()
        if not np.any(mask_tile):
            continue
        # DAB-H conversion (tile-based)
        lr_hed = rgb2hed(tile)
        lr_dab = butterworth(lr_hed[..., 2], cutoff_frequency_ratio=.02)
        lr_hem = lr_hed[..., 0]
        dab = lr_dab
        hem = lr_hem
        # HSV processing for artifact removal
        hsvt = rgb2hsv(tile)
        dif = hsvt[..., 1] - np.sum(tile, axis=2)
        try:
            _, dif_thr = threshold_multiotsu(dif)
        except Exception:
            dif_thr = 10000
        difbin = dif > dif_thr
        difbin_e = binary_erosion(difbin)
        difbin_be = remove_small_objects(difbin_e, min_size=680, connectivity=1)
        difbin_bed = binary_dilation(difbin_be)
        dont_analyze = filter_shapes_by_overlap(difbin, difbin_bed).astype(bool)
        # Process DAB channel
        dabbin = (dab > .08) & (~dont_analyze)
        dabbinb = remove_small_objects(dabbin, min_size=4)
        lobjsdab = remove_small_objects(dabbinb, min_size=70)
        dabbinm = dabbinb ^ lobjsdab
        clumpycellsdab = remove_small_objects(dabbinm, min_size=35)
        smallcellsdab = dabbinm ^ clumpycellsdab
        clumpycellsdabero = binary_erosion(clumpycellsdab)
        dabbinmselero = smallcellsdab + clumpycellsdabero
        # Process HEM channel
        hembin = (hem > .015) & (~dont_analyze)
        hembinb = remove_small_objects(hembin, min_size=4)
        lobjshem = remove_small_objects(hembinb, min_size=60)
        hembinm = hembinb ^ lobjshem
        clumpycellshem = remove_small_objects(hembinm, min_size=35)
        smallcellshem = hembinm ^ clumpycellshem
        clumpycellshemero = binary_erosion(clumpycellshem)
        hembinmselero = smallcellshem + clumpycellshemero
        # Combine DAB and HEM channels and process
        combbinm = hembinm + dabbinm
        clumpycells = remove_small_objects(combbinm, min_size=35)
        smallcells = combbinm ^ clumpycells
        clumpycellsero = binary_erosion(clumpycells)
        combbinmselero = smallcells + clumpycellsero
        mask_arr[x, y] = 1
        # Label connected components in DAB-H positive, HEM negative, and all cells
        _, polydabmselero = label(dabbinmselero, connectivity=1, return_num=True)
        _, polyhemmselero = label(hembinmselero, connectivity=1, return_num=True)
        _, polycommselero = label(combbinmselero, connectivity=1, return_num=True)
        # Store results in summary arrays
        cell[x, y] = polycommselero
        dabh[x, y] = polydabmselero
        hemh[x, y] = polyhemmselero
        # Calculate ratio of DAB-H positive cells to all cells
        if polydabmselero > polycommselero:
            rat[x, y] = 1
        elif polycommselero == 0:
            rat[x, y] = 0
        else:
            rat[x, y] = polydabmselero / polycommselero
        # Explicitly delete large intermediates
        del tile, dab, hem, lr_hed, lr_dab, lr_hem, hsvt, dif, difbin, difbin_e, difbin_be, difbin_bed, dont_analyze
        del dabbin, dabbinb, lobjsdab, dabbinm, clumpycellsdab, smallcellsdab, clumpycellsdabero, dabbinmselero
        del hembin, hembinb, lobjshem, hembinm, clumpycellshem, smallcellshem, clumpycellshemero, hembinmselero
        del combbinm, clumpycells, smallcells, clumpycellsero, combbinmselero, mask_tile
    return cell, dabh, hemh, rat, mask_arr

# New DABLR function 
def DABLR_V2(large_recon_pv, mask=None, down_factor = 10000):   
    # extract LR and 
    lr = large_recon_pv.numpy()                                 
    tiles = get_tile_list(lr, down_factor)
    # extract tissue mask
    if type(mask) == type(None):
        lr_mask, _ = get_LR_tissue_mask(large_recon_pv)
        lr_mask = lr_mask.numpy()
    else:
        lr_mask = mask
    # convert to DAB-H space and highpass filter
    lr_hed = rgb2hed(lr)
    lr_dab = butterworth(lr_hed[...,2], cutoff_frequency_ratio=.02)
    # lr_hem = butterworth(lr_hed[...,0], cutoff_frequency_ratio=.02)
    lr_hem = lr_hed[...,0]  # HEM channel
    ((_,_,_,_),(max_x,max_y)) = tiles[-1]
    cell = np.zeros((max_x+1,max_y+1))
    dabh = np.zeros((max_x+1,max_y+1))
    rat = np.zeros((max_x+1,max_y+1))
    hemh = np.zeros((max_x+1,max_y+1))
    mask = np.zeros((max_x+1,max_y+1))
    # Loop through tiles and process each one
    for ((start_x,start_y,end_x,end_y),(x,y)) in tqdm.tqdm(tiles):
        tile = lr[start_x:end_x, start_y:end_y,:]           # get tile
        dab = lr_dab[start_x:end_x, start_y:end_y]          # get DAB channel
        hem = lr_hem[start_x:end_x, start_y:end_y]          # get HEM channel
        mask_tile = lr_mask[start_x:end_x, start_y:end_y]   # get mask
        if not np.any(mask_tile):                           # if no tissue in tile, skip
            continue
        
        # HSV processing for artifact removal
        hsvt = rgb2hsv(tile)                                # convert tile to HSV space
        dif = hsvt[...,1] - np.sum(tile, axis=2)            # calculate difference between saturation and intensity
        try:
            _, dif_thr = threshold_multiotsu(dif)           # find threshold for artifact detection
        except:
            dif_thr = 10000                                 # if thresholding fails, set a high value to avoid artifacts
        # identify artifacts
        difbin = dif > dif_thr                              
        difbin_e = binary_erosion(difbin)                   
        difbin_be = remove_small_objects(difbin_e, min_size=680, connectivity=1) 
        difbin_bed = binary_dilation(difbin_be)
        dont_analyze = filter_shapes_by_overlap(difbin, difbin_bed).astype(bool) 
        # Process DAB channel
        dabbin = dab > .08                                              # threshold DAB channel
        dabbin = dabbin * np.invert(dont_analyze)                       # exclude artifacts
        
        ## Remove small objects and lobular objects
        dabbinb = remove_small_objects(dabbin, min_size=4)              
        lobjsdab = remove_small_objects(dabbinb, min_size=70)           
        dabbinm = dabbinb ^ lobjsdab
        ## Identify clumpy cells and small cells, then apply erosion 
        clumpycellsdab = remove_small_objects(dabbinm, min_size=35)     
        smallcellsdab = dabbinm ^ clumpycellsdab                        
        clumpycellsdabero = binary_erosion(clumpycellsdab)        
        dabbinmselero = smallcellsdab + clumpycellsdabero               # important step to get DAB-H positive cells
        # Process HEM channel
        hembin = (hem > .015)                                           # threshold HEM channel
        hembin = hembin * np.invert(dont_analyze)                       # exclude artifacts
        ## Remove small objects and lobular objects
        hembinb = remove_small_objects(hembin, min_size=4)        
        lobjshem = remove_small_objects(hembinb, min_size=60)
        hembinm = hembinb ^ lobjshem
        ## Identify clumpy cells and small cells, then apply erosion 
        clumpycellshem = remove_small_objects(hembinm, min_size=35)     
        smallcellshem = hembinm ^ clumpycellshem                        
        clumpycellshemero = binary_erosion(clumpycellshem)        
        hembinmselero = smallcellshem + clumpycellshemero               # important step to get DAB-H negative cells
        # Combine DAB and HEM channels and process
        combbinm = hembinm + dabbinm
        ## Identify clumpy cells and small cells, then apply erosion
        clumpycells = remove_small_objects(combbinm, min_size=35)
        smallcells = combbinm ^ clumpycells
        clumpycellsero = binary_erosion(clumpycells)
        combbinmselero = smallcells + clumpycellsero                    # important step to get all cells
        mask[x,y] = 1                                                   # mark tile as processed
        _, polydabmselero = label(dabbinmselero, connectivity=1 ,return_num=True)   # label DAB-H positive cells
        _, polyhemmselero = label(hembinmselero, connectivity=1 ,return_num=True)   # label DAB-H negative cells
        _, polycommselero = label(combbinmselero, connectivity=1, return_num=True)  # label all cells
        cell[x,y] = polycommselero 
        dabh[x,y] = polydabmselero
        hemh[x,y] = polyhemmselero
        if polydabmselero > polycommselero:
            rat[x, y] = 1
        elif polycommselero == 0:
            rat[x, y] = 0
        else:
            rat[x, y] = polydabmselero / polycommselero
    return cell, dabh, hemh, rat, mask

# Original DABLR function from Benjamin Chao
def DABLR_OG(large_recon_pv, mask=None, down_factor = 10000):
    large_recon = large_recon_pv.numpy()
    tiles = get_tile_list(large_recon, down_factor)
    if type(mask) == type(None):
        mask_LR, _ = get_LR_tissue_mask(large_recon_pv)
        mask_LR = mask_LR.numpy()
    else:
        mask_LR = mask
    dabfull = rgb2hed(large_recon)
    #del large_recon
    highpassdabfull = butterworth(dabfull[...,2], cutoff_frequency_ratio=.02)
    ((_,_,_,_),(max_x,max_y)) = tiles[-1]
    cell = np.zeros((max_x+1,max_y+1))
    dabh = np.zeros((max_x+1,max_y+1))
    rat = np.zeros((max_x+1,max_y+1))
    mask = np.zeros((max_x+1,max_y+1))
    for ((start_x,start_y,end_x,end_y),(x,y)) in tqdm.tqdm(tiles):
        tile = large_recon[start_x:end_x, start_y:end_y,:]
        dab = highpassdabfull[start_x:end_x, start_y:end_y]
        mask_tile = mask_LR[start_x:end_x, start_y:end_y]
        if not np.any(mask_tile):
            continue
        dabbin = dab > .08
        dabbinb = remove_small_objects(dabbin, min_size=4)
        hsvt = rgb2hsv(tile)
        dif = hsvt[...,1] - np.sum(tile, axis=2)
        try:
            _, dif_thr = threshold_multiotsu(dif)
        except:
            dif_thr = 10000
        difbin = dif > dif_thr
        difbin_e = binary_erosion(difbin)
        difbin_be = remove_small_objects(difbin_e, min_size=680, connectivity=1)
        difbin_bed = binary_dilation(difbin_be)
        dont_analyze = filter_shapes_by_overlap(difbin, difbin_bed).astype(bool)
        dabbinb = dabbinb * np.invert(dont_analyze)
        mask[x,y] = 1
        hem = dabfull[start_x:end_x, start_y:end_y,0]
        hemthresh = .015
        hembin = (hem > hemthresh)
        hembin = hembin * np.invert(dont_analyze)
        hembinb = remove_small_objects(hembin, min_size=4)
        lobjshem = remove_small_objects(hembinb, min_size=60)
        hembinm = hembinb ^ lobjshem
        lobjsdab = remove_small_objects(dabbinb, min_size=70)
        dabbinm = dabbinb ^ lobjsdab
        clumpycellsdab = remove_small_objects(dabbinm, min_size=35)
        smallcellsdab = dabbinm ^ clumpycellsdab
        clumpycellsdabero = binary_erosion(clumpycellsdab)
        dabbinmselero = smallcellsdab + clumpycellsdabero
        combbinm = hembinm + dabbinm
        clumpycells = remove_small_objects(combbinm, min_size=35)
        smallcells = combbinm ^ clumpycells
        clumpycellsero = binary_erosion(clumpycells)
        combbinmselero = smallcells + clumpycellsero
        _, polydabmselero = label(dabbinmselero, connectivity=1 ,return_num=True)
        _, polycommselero = label(combbinmselero, connectivity=1, return_num=True)
        cell[x,y] = polycommselero
        dabh[x,y] = polydabmselero
        if polydabmselero > polycommselero:
            rat[x, y] = 1
        elif polycommselero == 0:
            rat[x, y] = 0
        else:
            rat[x, y] = polydabmselero / polycommselero
    return cell, dabh, rat, mask
