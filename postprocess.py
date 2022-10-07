from skimage import morphology
import numpy as np
import skimage

# ==================================================
# ==================================================
def remove_small_objects(seg):
    '''
    Removes island from a binary image. Only keeps the largest island.
    '''
    processed = morphology.remove_small_objects(seg.astype(bool),
                                                min_size = 64,
                                                connectivity = 1).astype(float)
    return processed

# ==================================================
# ==================================================
def remove_islands(seg):
    '''
    
    '''
    res, N = skimage.measure.label(seg.astype(bool), return_num=True)
    res = res.astype(float)
    len_island = 0
    len_idx = 1
    if N > 1:
        for i in range(1,N+1):
            l = len(np.where(res.flatten()==i)[0])
            if l>=len_island:
                len_island = l
                len_idx = i
        res[np.where(res!=len_idx)] = 0

    return res, N

# ==================================================
# ==================================================
def postprocess_segmentation(seg):
    cleaned_seg = remove_small_objects(seg = seg)
    cleaned_seg, _ = remove_islands(seg = cleaned_seg)
    return cleaned_seg