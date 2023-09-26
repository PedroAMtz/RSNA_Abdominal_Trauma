
from scipy.ndimage import zoom
import numpy as np

def change_depth_siz(patient_volume: np.ndarray, target_depth: int=64) -> np.ndarray:

    """_Change the depth of an input volume as a numpy array
        considering SIZ algorithm and a desired target depth_

    Returns
    -------
    _np.ndarray_
        _Volume reduced from the original input volume containing
         the target depth as the total number of slices per volume_
    """
    current_depth = patient_volume.shape[-1]
    depth = current_depth / target_depth
    depth_factor = 1 / depth
    img_new = zoom(patient_volume, (1, 1, depth_factor), mode='nearest')
    return img_new