# This file will contain the functions used in the LITT project.
# These scripts were initially developed by Benjamin Chao in Dr. LaViolette's lab.
# They have been modified by Pouya Mirzaei.

import numpy as np
import skimage as ski
import histomicstk as htk
import tqdm
import pyvips
from skimage.filters import threshold_multiotsu, butterworth
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv, rgb2hed
from valis.preprocessing import create_tissue_mask_with_jc_dist

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

def DABLR_CELL(large_recon_pv, mask=None, down_factor = 1000):
    # extract LR and 
    lr = large_recon_pv.numpy()                                 
    tiles = get_tile_list(lr, down_factor)
    
    # get tissue mask
    if type(mask) == type(None):                    
        mask_LR, _ = get_LR_tissue_mask(large_recon_pv)
        mask_LR = mask_LR.numpy()
    else:
        mask_LR = mask

    # convert to DAB-H space and highpass filter
    lr_hed = rgb2hed(lr)
    lr_dab = butterworth(lr_hed[...,2], cutoff_frequency_ratio=.02)
    lr_hem = butterworth(lr_hed[...,0], cutoff_frequency_ratio=.02)

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

# Original DABLR function from Benjamin Chao
def DABLR(large_recon_pv, mask=None, down_factor = 10000):
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