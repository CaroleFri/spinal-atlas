#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:53:16 2026

@author: frindel
"""

from pathlib import Path

# ============================================================
# PARAMETERS TO EDIT
# ============================================================

ROOT = Path("/Users/frindel/Documents/Données-Corentin")
UTILS = ROOT / "utils"

REFERENCE_SUBJECT = "sub-28"

SUBJECTS = [
    "sub-27",
    "sub-29",
]

COMPONENTS = ["txx", "txy", "txz", "tyy", "tyz", "tzz"]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def subj_dir(subject):
    return ROOT / subject


def corrected_nii(subject):
    return f"{subject}_Corrected.nii.gz"


def corrected_bval(subject):
    return f"{subject}_Corrected.bval"


def corrected_bvec(subject):
    return f"{subject}_Corrected.bvec"


def b0_file(subject):
    return f"{subject}_b0.nii.gz"


def tensor_components_dir(subject):
    return f"{subject}_tensor_components"


def tensor_5d_file(subject):
    return f"{subject}_tensor_components/{subject}_tensor_5d.nii.gz"


def registration_dir(moving_subject, fixed_subject):
    return f"b0_registration_{moving_subject}_to_{fixed_subject}"


def registration_prefix(moving_subject, fixed_subject):
    return f"{moving_subject}_to_{fixed_subject}_"


# ============================================================
# PIPELINE
# ============================================================

for subject in SUBJECTS:

    print(f"\n=== Traitement de {subject} ===")

    # --------------------------------------------------------
    # 1. Extract b0 images
    # --------------------------------------------------------

    runfile(
        str(UTILS / "extract_b0.py"),
        args=(
            f"--dwi {corrected_nii(subject)} "
            f"--bval {corrected_bval(subject)} "
            f"--bvec {corrected_bvec(subject)} "
            f"--out {b0_file(subject)}"
        ),
        wdir=str(subj_dir(subject))
    )

    # --------------------------------------------------------
    # 2. Convert tensor components to a 5D file
    # --------------------------------------------------------

    runfile(
        str(UTILS / "convert_components_to_tensor5d.py"),
        args=(
            f"--subject {subject} "
            f"--tensor-root {tensor_components_dir(subject)} "
            f"--component-pattern {{subject}}_Corrected_{{component}}.nii.gz "
            f"--out {tensor_5d_file(subject)}"
        ),
        wdir=str(subj_dir(subject))
    )

    # --------------------------------------------------------
    # 3. Compute deformation from b0
    # --------------------------------------------------------

    runfile(
        str(UTILS / "register_b0_only.py"),
        args=(
            f"--fixed-b0 ../{REFERENCE_SUBJECT}/{b0_file(REFERENCE_SUBJECT)} "
            f"--moving-b0 {b0_file(subject)} "
            f"--outdir {registration_dir(subject, REFERENCE_SUBJECT)} "
            f"--prefix {registration_prefix(subject, REFERENCE_SUBJECT)}"
        ),
        wdir=str(subj_dir(subject))
    )

    # --------------------------------------------------------
    # 4. Apply deformation to the tensor
    # --------------------------------------------------------

    reg_dir = registration_dir(subject, REFERENCE_SUBJECT)
    reg_prefix = registration_prefix(subject, REFERENCE_SUBJECT)

    runfile(
        str(UTILS / "apply_existing_transforms_to_tensor.py"),
        args=(
            f"--tensor {tensor_5d_file(subject)} "
            f"--reference ../{REFERENCE_SUBJECT}/{b0_file(REFERENCE_SUBJECT)} "
            f"--warp {reg_dir}/{reg_prefix}1Warp.nii.gz "
            f"--affine {reg_dir}/{reg_prefix}0GenericAffine.mat "
            f"--out ../atlas/{subject}_tensor_to_{REFERENCE_SUBJECT}_reoriented.nii.gz "
            f"--split "
            f"--split-prefix {subject}_to_{REFERENCE_SUBJECT}"
        ),
        wdir=str(subj_dir(subject))
    )

    # --------------------------------------------------------
    # 5. Convert raw tensors to .fib files
    # --------------------------------------------------------

    runfile(
        str(UTILS / "convert_tensor_to_fib.py"),
        args=(
            f"--tensor {tensor_5d_file(subject)} "
            f"--template-fib ../{REFERENCE_SUBJECT}/{REFERENCE_SUBJECT}_Corrected.fib.gz "
            f"--out {tensor_components_dir(subject)}/{subject}_tensor_5d.fib.gz "
            f"--index-base zero"
        ),
        wdir=str(subj_dir(subject))
    )
    

    # --------------------------------------------------------
    # 6. Convert reoriented tensors to .fib files
    # --------------------------------------------------------

    runfile(
        str(UTILS / "convert_tensor_to_fib.py"),
        args=(
            f"--tensor ../atlas/{subject}_tensor_to_{REFERENCE_SUBJECT}_reoriented.nii.gz "
            f"--template-fib ../{REFERENCE_SUBJECT}/{REFERENCE_SUBJECT}_Corrected.fib.gz "
            f"--out ../atlas/{subject}_tensor_to_{REFERENCE_SUBJECT}_reoriented.fib.gz "
            f"--index-base zero"
        ),
        wdir=str(subj_dir(subject))
    )


# --------------------------------------------------------
# 7. Tensor fusion
# --------------------------------------------------------

runfile(
    str(UTILS / "fuse_tensors.py"),
    args=(
        f"--tensor-dir {ROOT / 'atlas'} "
        f"--outdir atlas "
        f"--pattern *_tensor_to_{REFERENCE_SUBJECT}_reoriented.nii.gz"
    ),
    wdir=str(ROOT)
)


# --------------------------------------------------------
# 8. Convert atlas files to .fib files
# --------------------------------------------------------

runfile(
    str(UTILS / "convert_tensor_to_fib.py"),
    args=(
        f"--tensor atlas/atlas_eigen_median_tensor_5d.nii.gz "
        f"--template-fib {REFERENCE_SUBJECT}/{REFERENCE_SUBJECT}_Corrected.fib.gz "
        f"--out atlas/atlas_eigen_median_tensor_5d.fib.gz "
        f"--index-base zero"
    ),
    wdir=str(ROOT)
)

runfile(
    str(UTILS / "convert_tensor_to_fib.py"),
    args=(
        f"--tensor atlas/atlas_logeuclidean_mean_tensor_5d.nii.gz "
        f"--template-fib {REFERENCE_SUBJECT}/{REFERENCE_SUBJECT}_Corrected.fib.gz "
        f"--out atlas/atlas_logeuclidean_mean_tensor_5d.fib.gz "
        f"--index-base zero"
    ),
    wdir=str(ROOT)
)