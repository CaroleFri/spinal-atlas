#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:05:14 2026

@author: frindel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:29:42 2026

@author: frindel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply existing ANTs transforms to a 5D diffusion tensor image.

Input:
    - tensor 5D NIfTI: X,Y,Z,1,6
    - reference image used with -r, usually fixed_b0.nii.gz
    - affine transform: 0GenericAffine.mat
    - warp field: 1Warp.nii.gz

Output:
    - warped/reoriented tensor 5D
    - optional split scalar components
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


# ANTs/ITK symmetric tensor order:
# [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
ANTS_INDEX_TO_COMPONENT = {
    0: "txx",
    1: "txy",
    2: "tyy",
    3: "txz",
    4: "tyz",
    5: "tzz",
}


def run(cmd, dry_run=False):
    print("\n" + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path.resolve()}")
    return path


def add_ants_to_path(ants_bin=None):
    if ants_bin:
        os.environ["PATH"] = str(Path(ants_bin)) + os.pathsep + os.environ["PATH"]

    ants_apply = shutil.which("antsApplyTransforms")

    if ants_apply is None:
        raise RuntimeError(
            "antsApplyTransforms not found. "
            "Use --ants-bin /path/to/ants/bin"
        )

    print(f"antsApplyTransforms: {ants_apply}")


def set_qform_sform_like(out_img, ref_img):
    out_img.set_qform(
        ref_img.get_qform(),
        code=int(ref_img.header["qform_code"]),
    )
    out_img.set_sform(
        ref_img.get_sform(),
        code=int(ref_img.header["sform_code"]),
    )
    return out_img


def check_tensor_5d(tensor_path):
    img = nib.load(str(tensor_path))
    data_shape = img.shape

    if len(data_shape) != 5:
        raise ValueError(
            f"Tensor image must be 5D, got shape: {data_shape}"
        )

    if data_shape[3] != 1 or data_shape[4] != 6:
        raise ValueError(
            f"Expected tensor shape X,Y,Z,1,6, got: {data_shape}"
        )

    print(f"Input tensor shape: {data_shape}")


def split_tensor_components(tensor_path, outdir, prefix="tensor_reoriented"):
    """
    Split warped/reoriented 5D tensor into 6 scalar NIfTI components.

    Input tensor order is assumed to be ANTs/ITK:
        [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
    """

    tensor_path = Path(tensor_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(tensor_path))
    data = img.get_fdata(dtype=np.float64)

    if data.ndim != 5 or data.shape[3] != 1 or data.shape[4] != 6:
        raise ValueError(
            f"Unexpected tensor shape: {data.shape}. "
            "Expected X,Y,Z,1,6."
        )

    for index, comp in ANTS_INDEX_TO_COMPONENT.items():
        arr = data[..., 0, index].astype(np.float32)

        out_path = outdir / f"{prefix}_{comp}.nii.gz"

        out_img = nib.Nifti1Image(
            arr,
            affine=img.affine,
            header=img.header.copy(),
        )
        out_img = set_qform_sform_like(out_img, img)
        out_img.header.set_data_dtype(np.float32)

        nib.save(out_img, str(out_path))
        print(f"Saved component: {out_path.resolve()}")


def apply_transforms_to_tensor(
    tensor,
    reference,
    warp,
    affine,
    output,
    ants_bin=None,
    split=False,
    split_prefix="tensor_reoriented",
    dry_run=False,
):
    """
    Apply existing ANTs transforms to a diffusion tensor image.

    The transform direction is assumed to be:
        moving tensor space -> reference/fixed space

    Transform order for ANTs:
        -t warp
        -t affine
    """

    add_ants_to_path(ants_bin)

    tensor = require_file(tensor, "input tensor 5D")
    reference = require_file(reference, "reference image")
    warp = require_file(warp, "warp field")
    affine = require_file(affine, "affine transform")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    check_tensor_5d(tensor)

    run(
        [
            "antsApplyTransforms",
            "-d", "3",
            "-e", "2",
            "-i", tensor,
            "-r", reference,
            "-o", output,
            "-t", warp,
            "-t", affine,
        ],
        dry_run=dry_run,
    )

    if not dry_run:
        require_file(output, "warped/reoriented tensor")

        if split:
            split_tensor_components(
                tensor_path=output,
                outdir=output.parent,
                prefix=split_prefix,
            )

    print("\nDone.")
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply existing ANTs transforms to a 5D diffusion tensor."
    )

    parser.add_argument(
        "--tensor",
        required=True,
        help="Input 5D tensor NIfTI, shape X,Y,Z,1,6.",
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Reference image for -r, usually fixed_b0.nii.gz.",
    )

    parser.add_argument(
        "--warp",
        required=True,
        help="ANTs warp field, e.g. 1Warp.nii.gz.",
    )

    parser.add_argument(
        "--affine",
        required=True,
        help="ANTs affine matrix, e.g. 0GenericAffine.mat.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output warped/reoriented 5D tensor .nii.gz.",
    )

    parser.add_argument(
        "--ants-bin",
        default="/opt/miniconda3/envs/ants_tensor/bin",
        help="Directory containing antsApplyTransforms.",
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Also split output tensor into 6 scalar components.",
    )

    parser.add_argument(
        "--split-prefix",
        default="tensor_reoriented",
        help="Prefix for split component outputs.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    apply_transforms_to_tensor(
        tensor=args.tensor,
        reference=args.reference,
        warp=args.warp,
        affine=args.affine,
        output=args.out,
        ants_bin=args.ants_bin,
        split=args.split,
        split_prefix=args.split_prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()