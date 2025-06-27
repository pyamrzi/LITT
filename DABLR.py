import numpy as np
import skimage as ski
import histomicstk as htk
from skimage.filters import threshold_multiotsu, butterworth
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation
import tqdm as tqdm
from get_LR_tissue_mask import get_LR_tissue_mask
from get_tile_list import get_tile_list
from skimage.color import rgb2hsv, rgb2hed

import numpy as np
from skimage.measure import label, regionprops

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
            