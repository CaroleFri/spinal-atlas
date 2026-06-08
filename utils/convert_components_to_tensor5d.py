#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:08:17 2026

@author: frindel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convertit 6 composantes tensorielle NIfTI en tenseur 5D NIfTI.

Entrée :
    txx, txy, txz, tyy, tyz, tzz

Sortie :
    tensor_5d.nii.gz de forme X,Y,Z,1,6

Aucune transformation spatiale n'est appliquée.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


COMPONENTS = ["txx", "txy", "txz", "tyy", "tyz", "tzz"]

# Ordre ITK/ANTs pour tenseur symétrique 3D :
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
    Trouve le chemin d'une composante tensorielle.

    component_pattern peut être par exemple :
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
            f"Composante introuvable pour {component}: {path.resolve()}"
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
    Convertit 6 composantes tensorielle en image tenseur 5D.

    Paramètres
    ----------
    subject : str
        Identifiant sujet, par exemple "sub-28".

    tensor_root : str or Path
        Dossier racine contenant les composantes.

    out_path : str or Path
        Chemin de sortie du tenseur 5D .nii.gz.

    component_pattern : str
        Pattern relatif à tensor_root.
        Doit contenir {subject} et {component}.

    component_to_index : dict or None
        Mapping vers l'ordre de stockage du tenseur 5D.
        Par défaut : ordre ANTs/ITK [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz].

    dtype : np.dtype
        Type numérique de sortie.

    verbose : bool
        Affiche les fichiers utilisés.

    Retour
    ------
    out_path : Path
        Chemin du tenseur 5D généré.
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

    # Force explicitement le layout 5D : X,Y,Z,1,6
    hdr["dim"][0] = 5
    hdr["dim"][1] = shape[0]
    hdr["dim"][2] = shape[1]
    hdr["dim"][3] = shape[2]
    hdr["dim"][4] = 1
    hdr["dim"][5] = 6
    hdr["dim"][6] = 1
    hdr["dim"][7] = 1

    # NIFTI_INTENT_SYMMATRIX = 1005, matrice symétrique 3x3
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
        help="Sujet, ex: sub-28",
    )

    parser.add_argument(
        "--tensor-root",
        default=".",
        help="Dossier racine contenant les composantes.",
    )

    parser.add_argument(
        "--component-pattern",
        default="{subject}_tensor_components/{subject}_Corrected_{component}.nii.gz",
        help=(
            "Pattern relatif à tensor-root. "
            "Champs disponibles: {subject}, {component}."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Chemin de sortie, ex: sub-28_tensor_5d.nii.gz",
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