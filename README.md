# npview: Python module for batch visualization of 2D numpy arrays and 1D slices

## Features
- Batch imshow for multiple 2D numpy arrays
- Flexible 1D slicing via string spec (horizontal/vertical, range, index)
- Overlay slice lines on images
- Overlay all 1D slices in extra column (diff mode)
- Overlay contours for all images in extra column (diff mode)
- Custom colormap, auto vmin/vmax
- CLI and Python API

## API

### plot_images_and_slices
```python
plot_images_and_slices(arrs, slices, save_path=None, cmap='jet', diff=False, contour_levels=None)
```
- `arrs`: list of 2D numpy arrays
- `slices`: list of slice dicts (from `parse_slice`)
- `save_path`: path to save figure (if None, show interactively)
- `cmap`: colormap for imshow
- `diff`: if True, overlay all 1D plots and contours in extra column
- `contour_levels`: list of contour levels for each array

### parse_slice
```python
parse_slice(slice_str, arr_shape)
```
- `slice_str`: e.g. 'h,10:20:at15' or 'v,5:30:at20'
- `arr_shape`: shape of numpy array
- Returns: dict with keys: type ('h' or 'v'), start, end, at

### load_npy_files
```python
load_npy_files(file_list)
```
- `file_list`: list of .npy file paths
- Returns: list of numpy arrays

## CLI Example
```bash
python examples/plot_numpy_images.py test1.npy test2.npy --slice "h,10:20:at15" "v,5:30:at20" --diff --contour 0.2,0.3 --cmap viridis --save-figure out.png
```

## Unit Test
Run `python tests/test_npview.py` to verify all features.
