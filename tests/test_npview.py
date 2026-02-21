import numpy as np
import os
from npview import plot_images_and_slices, parse_slice, load_npy_files

def create_test_files():
    arr1 = np.random.rand(30, 40)
    arr2 = np.random.rand(30, 40)
    np.save('test1.npy', arr1)
    np.save('test2.npy', arr2)
    print('Test .npy files created: test1.npy, test2.npy')

def test_basic():
    arrs = load_npy_files(['test1.npy', 'test2.npy'])
    arr_shape = arrs[0].shape
    slices = [parse_slice("h,10:20:at15", arr_shape), parse_slice("v,5:30:at20", arr_shape)]
    plot_images_and_slices(arrs, slices, save_path='unit_test_out.png')
    assert os.path.exists('unit_test_out.png')

def test_cmap():
    arrs = load_npy_files(['test1.npy', 'test2.npy'])
    arr_shape = arrs[0].shape
    slices = [parse_slice("h,10:20:at15", arr_shape)]
    plot_images_and_slices(arrs, slices, save_path='unit_test_cmap.png', cmap='plasma')
    assert os.path.exists('unit_test_cmap.png')

def test_diff_contour():
    arrs = load_npy_files(['test1.npy', 'test2.npy'])
    arr_shape = arrs[0].shape
    slices = [parse_slice("h,10:20:at15", arr_shape), parse_slice("v,5:30:at20", arr_shape)]
    plot_images_and_slices(arrs, slices, save_path='unit_test_diff.png', diff=True, contour_levels=[0.2, 0.3])
    assert os.path.exists('unit_test_diff.png')

def run_all():
    create_test_files()
    test_basic()
    test_cmap()
    test_diff_contour()
    print('All unit tests passed.')

if __name__ == "__main__":
    run_all()
