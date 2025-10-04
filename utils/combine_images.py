import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_images(image_paths, output_image_name=None, cols=3, figsize=(12, 8), show=True):
    """
    Display multiple images together in a grid and optionally save the figure.

    Args:
        image_paths (list[str]): List of image file paths.
        output_image_name (str): Path to save the combined image (optional).
        cols (int): Number of columns in the figure grid.
        figsize (tuple): Size of the figure (width, height).
        show (bool): Whether to display the figure interactively.
    """
    num_images = len(image_paths)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=figsize)

    for i, img_path in enumerate(image_paths):
        try:
            img = mpimg.imread(img_path)
        except Exception as e:
            print(f"❌ Could not read {img_path}: {e}")
            continue

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")  # hide axes and titles

    plt.tight_layout(pad=0.2)

    if output_image_name:
        plt.savefig(output_image_name, bbox_inches='tight', pad_inches=0)
        print(f"✅ Saved combined image to: {output_image_name}")

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Combine multiple images into one figure.")
    parser.add_argument(
        "--input_images",
        nargs="+",
        required=True,
        help="List of input image paths (space-separated)."
    )
    parser.add_argument(
        "--output_image_name",
        required=True,
        help="Output file path for the combined image."
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in the image grid (default: 3)."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12,
        help="Width of the figure in inches."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=8,
        help="Height of the figure in inches."
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the figure (useful for batch scripts)."
    )

    args = parser.parse_args()

    show_images(
        image_paths=args.input_images,
        output_image_name=args.output_image_name,
        cols=args.cols,
        figsize=(args.width, args.height),
        show=not args.no_show
    )

if __name__ == "__main__":
    main()
