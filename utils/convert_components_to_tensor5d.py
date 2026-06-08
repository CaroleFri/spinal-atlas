#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:08:17 2026

@author: frindel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert 6 NIfTI tensor components into a 5D NIfTI tensor.

Input:
    txx, txy, txz, tyy, tyz, tzz

Output:
    tensor_5d.nii.gz de forme X,Y,Z,1,6

No spatial transformation is applied.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


COMPONENTS = ["txx", "txy", "txz", "tyy", "tyz", "tzz"]

# ITK/ANTs order for a 3D symmetric tensor:
# [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
COMPONENT_TO_ANTS_INDEX = {
    "txx": 0,
    "txy": 1,
    "tyy": 2,
    "txz": 3,
    "tyz": 4,
    "tzz": 5,
}


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


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float64)
    return img, data


def find_component_path(tensor_root, subject, component, component_pattern):
    """
    Find the path of a tensor component.

    component_pattern can be, for example:
        "{subject}_tensor_components/{subject}_Corrected_{component}.nii.gz"

    ou :
        "{subject}_Corrected_moelle_{component}.nii.gz"
    """

    tensor_root = Path(tensor_root)

    rel_path = component_pattern.format(
        subject=subject,
        component=component,
    )

    path = tensor_root / rel_path

    if not path.exists():
        raise FileNotFoundError(
            f"Component not found for {component}: {path.resolve()}"
        )

    return path


def convert_6_components_to_tensor_5d(
    subject,
    tensor_root,
    out_path,
    component_pattern="{subject}_tensor_components/{subject}_Corrected_{component}.nii.gz",
    component_to_index=None,
    dtype=np.float32,
    verbose=True,
):
    """
    Convert 6 tensor components into a 5D tensor image.

    Parameters
    ----------
    subject : str
        Subject identifier, for example "sub-28".

    tensor_root : str or Path
        Root directory containing the components.

    out_path : str or Path
        Output path for the 5D tensor .nii.gz file.

    component_pattern : str
        Pattern relative to tensor_root.
        Must contain {subject} and {component}.

    component_to_index : dict or None
        Mapping to the 5D tensor storage order.
        Default: ANTs/ITK order [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz].

    dtype : np.dtype
        Output numeric type.

    verbose : bool
        Print the files being used.

    Retour
    ------
    out_path : Path
        Path of the generated 5D tensor.
    """

    if component_to_index is None:
        component_to_index = COMPONENT_TO_ANTS_INDEX

    tensor_root = Path(tensor_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        comp: find_component_path(
            tensor_root=tensor_root,
            subject=subject,
            component=comp,
            component_pattern=component_pattern,
        )
        for comp in COMPONENTS
    }

    if verbose:
        print("\n=== COMPONENT PATHS ===")
        for comp, path in paths.items():
            print(f"{comp}: {path.resolve()}")
        print("=======================\n")

    ref_img, ref_data = load_nifti(paths["txx"])
    shape = ref_data.shape

    loaded = {"txx": ref_data}

    for comp in ["txy", "txz", "tyy", "tyz", "tzz"]:
        img, data = load_nifti(paths[comp])

        if data.shape != shape:
            raise ValueError(
                f"Shape mismatch pour {comp}: {data.shape}, attendu {shape}"
            )

        if not np.allclose(img.affine, ref_img.affine):
            raise ValueError(
                f"Affine différente pour {comp}. "
                "Les 6 composantes doivent être dans le même espace."
            )

        loaded[comp] = data

    tensor = np.zeros(shape + (1, 6), dtype=dtype)

    for comp, index in component_to_index.items():
        tensor[..., 0, index] = loaded[comp].astype(dtype)

    tensor_img = nib.Nifti1Image(
        tensor,
        affine=ref_img.affine,
        header=ref_img.header.copy(),
    )

    tensor_img = set_qform_sform_like(tensor_img, ref_img)

    hdr = tensor_img.header
    hdr.set_data_dtype(dtype)

    # Explicitly force the 5D layout: X,Y,Z,1,6
    hdr["dim"][0] = 5
    hdr["dim"][1] = shape[0]
    hdr["dim"][2] = shape[1]
    hdr["dim"][3] = shape[2]
    hdr["dim"][4] = 1
    hdr["dim"][5] = 6
    hdr["dim"][6] = 1
    hdr["dim"][7] = 1

    # NIFTI_INTENT_SYMMATRIX = 1005, symmetric 3x3 matrix
    hdr["intent_code"] = 1005
    hdr["intent_p1"] = 3

    nib.save(tensor_img, str(out_path))

    if verbose:
        print(f"Saved tensor 5D: {out_path.resolve()}")
        print(f"Output shape: {tensor.shape}")

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 6 tensor components to a 5D tensor NIfTI without spatial transformation."
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject, e.g. sub-28",
    )

    parser.add_argument(
        "--tensor-root",
        default=".",
        help="Root directory containing the components.",
    )

    parser.add_argument(
        "--component-pattern",
        default="{subject}_tensor_components/{subject}_Corrected_{component}.nii.gz",
        help=(
            "Pattern relative to tensor-root. "
            "Available fields: {subject}, {component}."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output path, e.g. sub-28_tensor_5d.nii.gz",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    convert_6_components_to_tensor_5d(
        subject=args.subject,
        tensor_root=args.tensor_root,
        component_pattern=args.component_pattern,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()