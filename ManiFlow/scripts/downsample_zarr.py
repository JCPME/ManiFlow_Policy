"""
Downsample image keys in a zarr dataset to a target resolution.
Creates a new zarr with all non-image keys copied as-is.

Usage:
    python scripts/downsample_zarr.py \
        --input data/coffee_cup_expert.zarr \
        --output data/coffee_cup_expert_224.zarr \
        --image-keys camera_1 camera_2 \
        --size 224
"""
import argparse
import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm


def resize_frames(arr, size):
    """Resize (N, H, W, C) uint8 array to (N, size, size, C)."""
    out = np.empty((len(arr), size, size, arr.shape[-1]), dtype=arr.dtype)
    for i, frame in enumerate(tqdm(arr, desc=f'  resizing', leave=False)):
        out[i] = np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-keys', nargs='+', default=['camera_1'])
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    src = zarr.open(args.input, 'r')
    dst = zarr.open(args.output, 'w')

    # copy meta as-is
    zarr.copy_all(src['meta'], dst.require_group('meta'))

    # copy data, resizing image keys
    dst_data = dst.require_group('data')
    for key in src['data']:
        arr = src['data'][key]
        print(f'Processing key: {key}  shape={arr.shape}')
        if key in args.image_keys:
            data = resize_frames(arr[:], args.size)
        else:
            data = arr[:]
        dst_data.array(key, data, chunks=arr.chunks, dtype=arr.dtype, overwrite=True)
        print(f'  -> saved shape={data.shape}')

    print(f'\nDone. Output: {args.output}')


if __name__ == '__main__':
    main()
