
"""
npview.plot
-------------
Batch visualization of 2D numpy arrays and 1D slices.
API:
    - plot_images_and_slices
    - parse_slice
    - load_npy_files
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt

def parse_slice(slice_str: str, arr_shape: tuple) -> dict:
    """
    Parse a slice string like 'h,10:20:at15' or 'v,5:30:at20'.
    Returns: dict with keys: type ('h' or 'v'), start, end, at
    """
    try:
        parts = slice_str.split(',')
        if len(parts) == 3:
            axis, rng, at = parts
        elif len(parts) == 2:
            axis, rest = parts
            if ':at' in rest:
                rng, at = rest.rsplit(':at', 1)
                at = 'at' + at
            else:
                raise ValueError(f"Invalid slice string: missing 'at' part")
        else:
            raise ValueError(f"Invalid slice string: {slice_str}")
        axis = axis.strip().lower()
        if axis not in ('h', 'v'):
            raise ValueError(f"Invalid axis: {axis}")
        rng, at_val = rng.strip(), at.strip()
        if not at_val.startswith('at'):
            raise ValueError(f"Invalid at: {at_val}")
        at_idx = int(at_val[2:])
        if ':' in rng:
            start, end = rng.split(':')
            start = int(start) if start else 0
            end = int(end) if end else (arr_shape[1] if axis == 'h' else arr_shape[0])
        else:
            start = int(rng) if rng else 0
            end = start + 1
        return {'type': axis, 'start': start, 'end': end, 'at': at_idx}
    except Exception as e:
        raise ValueError(f"Invalid slice string '{slice_str}': {e}")

def load_npy_files(file_list: list) -> list:
    arrs = []
    for f in file_list:
        arr = np.load(f)
        arrs.append(arr)
    shapes = [a.shape for a in arrs]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All input arrays must have the same shape, got: {shapes}")
    return arrs

def plot_images_and_slices(
    arrs: list,
    slices: list,
    save_path: str = None,
    cmap: str = 'jet',
    diff: bool = False,
    contour_levels: list = None
) -> None:
    """
    Plot 2D numpy images and 1D slices, with optional overlay and contour.
    Args:
        arrs: list of 2D numpy arrays
        slices: list of slice dicts
        save_path: path to save figure
        cmap: colormap for imshow
        diff: if True, overlay 1D plots and contours
        contour_levels: list of contour levels for each array
    """
    n_img = len(arrs)
    n_slice = len(slices)
    vmin = min(arr.min() for arr in arrs)
    vmax = max(arr.max() for arr in arrs)
    ncols = n_img + (1 if diff else 0)
    nrows = 1 + n_slice
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(4*ncols, 3*nrows))
    # First row: imshow
    for i, arr in enumerate(arrs):
        ax = axes[0, i]
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Image {i+1}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for slc in slices:
            if slc['type'] == 'h':
                y = slc['at']
                x0 = slc['start']
                x1 = slc['end'] - 1
                ax.plot([x0, x1], [y, y], color='red', linewidth=2)
            else:
                x = slc['at']
                y0 = slc['start']
                y1 = slc['end'] - 1
                ax.plot([x, x], [y0, y1], color='red', linewidth=2)
    # First row diff column: contour overlay
    if diff:
        ax = axes[0, n_img]
        for i, arr in enumerate(arrs):
            if contour_levels is not None:
                if i < len(contour_levels):
                    level = contour_levels[i]
                else:
                    level = contour_levels[-1]
            else:
                level = (arr.min() + arr.max()) / 2
            cs = ax.contour(arr, levels=[level], colors=[f'C{i}'], linewidths=2)
        # Place label at top right
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = xlim[1] - 0.02 * (xlim[1] - xlim[0])
        y = ylim[0] + 0.08 * (ylim[1] - ylim[0])
        for i in range(len(arrs)):
            ax.text(x, y + i*0.08*(ylim[1]-ylim[0]), f'Img{i+1}', color=f'C{i}', fontsize=10, weight='bold', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
        ax.set_title('Contours Overlay')
    # Slices
    for j, slc in enumerate(slices):
        for i, arr in enumerate(arrs):
            ax = axes[j+1, i]
            if slc['type'] == 'h':
                y = slc['at']
                if y < 0 or y >= arr.shape[0]:
                    raise IndexError(f"Row index {y} out of bounds for shape {arr.shape}")
                data = arr[y, slc['start']:slc['end']]
                ax.plot(np.arange(slc['start'], slc['end']), data, label=f'Img{i+1}')
                ax.set_title(f"Img{i+1} H: row {y}, {slc['start']}:{slc['end']}")
            else:
                x = slc['at']
                if x < 0 or x >= arr.shape[1]:
                    raise IndexError(f"Col index {x} out of bounds for shape {arr.shape}")
                data = arr[slc['start']:slc['end'], x]
                ax.plot(np.arange(slc['start'], slc['end']), data, label=f'Img{i+1}')
                ax.set_title(f"Img{i+1} V: col {x}, {slc['start']}:{slc['end']}")
            ax.legend()
        # diff column: overlay all 1D curves
        if diff:
            ax = axes[j+1, n_img]
            for i, arr in enumerate(arrs):
                if slc['type'] == 'h':
                    y = slc['at']
                    if y < 0 or y >= arr.shape[0]:
                        continue
                    data = arr[y, slc['start']:slc['end']]
                    ax.plot(np.arange(slc['start'], slc['end']), data, label=f'Img{i+1}')
                else:
                    x = slc['at']
                    if x < 0 or x >= arr.shape[1]:
                        continue
                    data = arr[slc['start']:slc['end'], x]
                    ax.plot(np.arange(slc['start'], slc['end']), data, label=f'Img{i+1}')
            ax.set_title('Overlay 1D')
            ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
