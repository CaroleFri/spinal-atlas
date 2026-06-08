#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract b0 volumes from a 4D diffusion NIfTI using a .bval file.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def read_bvals(path):
    """Read bvals from either one-row or one-column text file."""
    bvals = np.loadtxt(path)
    bvals = np.asarray(bvals).reshape(-1)
    return bvals


def save_3d_like(data, ref_img, out_path):
    """Save 3D data with affine/qform/sform copied from reference image."""
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)

    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, hdr)
    out.set_qform(ref_img.get_qform(), code=int(ref_img.header["qform_code"]))
    out.set_sform(ref_img.get_sform(), code=int(ref_img.header["sform_code"]))
    nib.save(out, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract b0 volume(s) from a 4D DWI NIfTI using bvals."
    )

    parser.add_argument(
        "--dwi",
        required=True,
        help="Input 4D DWI NIfTI, e.g. sub-27_Corrected.nii.gz",
    )
    parser.add_argument(
        "--bval",
        required=True,
        help="Input bval file associated with the DWI image.",
    )
    parser.add_argument(
        "--bvec",
        default=None,
        help="Optional bvec file. Not used for extraction, only checked if provided.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output 3D b0 NIfTI, e.g. sub-27_b0.nii.gz",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Volumes with bval <= threshold are considered b0. Default: 50.",
    )
    parser.add_argument(
        "--mode",
        choices=["mean", "first"],
        default="mean",
        help="How to combine b0 volumes. 'mean' averages all b0s, 'first' keeps first b0. Default: mean.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Manually extract this volume index instead of using bvals. 0-based indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected b0 indices and exit without writing output.",
    )

    args = parser.parse_args()

    dwi_path = Path(args.dwi)
    bval_path = Path(args.bval)
    out_path = Path(args.out)

    if not dwi_path.exists():
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")
    if not bval_path.exists():
        raise FileNotFoundError(f"bval file not found: {bval_path}")
    if args.bvec is not None and not Path(args.bvec).exists():
        raise FileNotFoundError(f"bvec file not found: {args.bvec}")

    img = nib.load(str(dwi_path))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 4:
        raise ValueError(f"Expected a 4D image, got shape {data.shape}")

    n_volumes = data.shape[3]
    bvals = read_bvals(str(bval_path))

    if len(bvals) != n_volumes:
        raise ValueError(
            f"Number of bvals ({len(bvals)}) does not match DWI volumes ({n_volumes})"
        )

    if args.bvec is not None:
        bvecs = np.loadtxt(args.bvec)
        if bvecs.shape not in [(n_volumes, 3), (3, n_volumes)]:
            raise ValueError(
                f"Unexpected bvec shape {bvecs.shape}; expected ({n_volumes}, 3) or (3, {n_volumes})"
            )
        print(f"bvec shape: {bvecs.shape}")

    if args.index is not None:
        if args.index < 0 or args.index >= n_volumes:
            raise ValueError(f"Index {args.index} outside range [0, {n_volumes - 1}]")
        b0_indices = np.array([args.index], dtype=int)
    else:
        b0_indices = np.where(bvals <= args.threshold)[0]

    if len(b0_indices) == 0:
        raise ValueError(
            f"No b0 volume found with threshold {args.threshold}. "
            "Try increasing --threshold or use --index."
        )

    print(f"DWI shape: {data.shape}")
    print(f"Number of volumes: {n_volumes}")
    print(f"b0 threshold: {args.threshold}")
    print(f"Detected b0 indices: {b0_indices.tolist()}")
    print(f"Detected b0 bvals: {bvals[b0_indices].tolist()}")

    if args.dry_run:
        print("Dry run: no output written.")
        return

    if args.mode == "first":
        b0 = data[..., b0_indices[0]]
        print(f"Using first b0 volume: index {b0_indices[0]}")
    else:
        b0 = np.mean(data[..., b0_indices], axis=3)
        print(f"Averaging {len(b0_indices)} b0 volume(s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_3d_like(b0, img, str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
