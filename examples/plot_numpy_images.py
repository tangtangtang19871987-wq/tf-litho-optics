import argparse
from npview import plot_images_and_slices, parse_slice, load_npy_files

def main():
    parser = argparse.ArgumentParser(description="Batch plot 2D numpy images and 1D slices.")
    parser.add_argument('files', nargs='+', help="Input .npy files (all must have same shape)")
    parser.add_argument('--slice', dest='slices', nargs='*', default=[], help="Slice definitions, e.g. 'h,10:20:at15'")
    parser.add_argument('--save-figure', dest='save_figure', default=None, help="Path to save the figure as PNG")
    parser.add_argument('--cmap', dest='cmap', default='jet', help="Colormap for imshow (default: jet)")
    parser.add_argument('--diff', action='store_true', help="Overlay all 1D plots and contours in extra column")
    parser.add_argument('--contour', dest='contour', default=None, help="Contour levels, e.g. 0.2 or 0.2,0.3,0.4")
    args = parser.parse_args()
    arrs = load_npy_files(args.files)
    arr_shape = arrs[0].shape
    slices = [parse_slice(s, arr_shape) for s in args.slices]
    contour_levels = [float(x) for x in args.contour.split(',')] if args.contour else None
    plot_images_and_slices(
        arrs, slices,
        save_path=args.save_figure,
        cmap=args.cmap,
        diff=args.diff,
        contour_levels=contour_levels
    )

if __name__ == "__main__":
    main()
