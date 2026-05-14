"""
Downsample image keys in a zarr dataset to a target resolution.
Creates a new zarr with all non-image keys copied as-is.
Processes image arrays in chunks to avoid loading all data into RAM.

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
import cv2
from tqdm import tqdm


def resize_key_chunked(src_arr, dst_data, key, size, chunk_size=500):
    N = len(src_arr)
    out = dst_data.zeros(
        name=key,
        shape=(N, size, size, src_arr.shape[-1]),
        chunks=(min(chunk_size, N), size, size, src_arr.shape[-1]),
        dtype=src_arr.dtype,
        overwrite=True,
    )
    for start in tqdm(range(0, N, chunk_size), desc=f'  {key}'):
        end = min(start + chunk_size, N)
        chunk = src_arr[start:end]  # (C, H, W, 3)
        resized = np.stack(
            [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
             for frame in chunk]
        )
        out[start:end] = resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-keys', nargs='+', default=['camera_1'])
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--chunk', type=int, default=500,
                        help='frames processed per chunk (controls peak RAM)')
    args = parser.parse_args()

    src = zarr.open(args.input, 'r')
    dst = zarr.open(args.output, 'w')

    zarr.copy_all(src['meta'], dst.require_group('meta'))

    dst_data = dst.require_group('data')
    for key in src['data']:
        arr = src['data'][key]
        print(f'Processing {key}  shape={arr.shape}')
        if key in args.image_keys:
            resize_key_chunked(arr, dst_data, key, args.size, args.chunk)
        else:
            dst_data.array(key, arr[:], chunks=arr.chunks, dtype=arr.dtype, overwrite=True)
        print(f'  -> done')

    print(f'\nDone. Output: {args.output}')


if __name__ == '__main__':
    main()
