#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:27:29 2026

@author: frindel
"""

#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib


# ============================================================
# Configuration
# ============================================================

EPS = 1e-6

# Axe vertical anatomique utilisé pour stabiliser les vecteurs propres
# À adapter si nécessaire selon ton espace image.
VERTICAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Ordre des composantes dans le tenseur 5D X,Y,Z,1,6
# Hypothèse ANTs classique : [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
COMPONENT_TO_ANTS_INDEX = {
    "txx": 0,
    "txy": 1,
    "tyy": 2,
    "txz": 3,
    "tyz": 4,
    "tzz": 5,
}

# ============================================================
# Fonctions utilitaires
# ============================================================

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)

    if n < eps:
        return np.zeros_like(v)

    return v / n


def make_spd(D, eps=EPS):
    """
    Force un tenseur symétrique défini positif.
    """

    D = 0.5 * (D + D.T)

    vals, vecs = np.linalg.eigh(D)
    vals = np.clip(vals, eps, None)

    D_spd = vecs @ np.diag(vals) @ vecs.T
    D_spd = 0.5 * (D_spd + D_spd.T)

    return D_spd


def tensor_log(D):
    vals, vecs = np.linalg.eigh(D)
    vals = np.clip(vals, EPS, None)

    return vecs @ np.diag(np.log(vals)) @ vecs.T


def tensor_exp(D):
    vals, vecs = np.linalg.eigh(D)

    return vecs @ np.diag(np.exp(vals)) @ vecs.T


# ============================================================
# Conversion tenseur 5D <-> matrice 3x3
# ============================================================

def tensor5d_to_matrix(tensor5d):
    """
    Convertit un tenseur 5D NIfTI X,Y,Z,1,6
    en tenseur matriciel X,Y,Z,3,3.
    """

    if tensor5d.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape {tensor5d.shape}")

    if tensor5d.shape[3] != 1:
        raise ValueError(f"Expected dim 4 = 1, got {tensor5d.shape[3]}")

    if tensor5d.shape[4] != 6:
        raise ValueError(f"Expected dim 5 = 6, got {tensor5d.shape[4]}")

    shape = tensor5d.shape[:3]
    out = np.zeros(shape + (3, 3), dtype=np.float64)

    t = tensor5d[..., 0, :]

    out[..., 0, 0] = t[..., COMPONENT_TO_ANTS_INDEX["txx"]]

    out[..., 0, 1] = t[..., COMPONENT_TO_ANTS_INDEX["txy"]]
    out[..., 1, 0] = t[..., COMPONENT_TO_ANTS_INDEX["txy"]]

    out[..., 0, 2] = t[..., COMPONENT_TO_ANTS_INDEX["txz"]]
    out[..., 2, 0] = t[..., COMPONENT_TO_ANTS_INDEX["txz"]]

    out[..., 1, 1] = t[..., COMPONENT_TO_ANTS_INDEX["tyy"]]

    out[..., 1, 2] = t[..., COMPONENT_TO_ANTS_INDEX["tyz"]]
    out[..., 2, 1] = t[..., COMPONENT_TO_ANTS_INDEX["tyz"]]

    out[..., 2, 2] = t[..., COMPONENT_TO_ANTS_INDEX["tzz"]]

    return out


def matrix_to_tensor5d(tensor_matrix):
    """
    Convertit un tenseur matriciel X,Y,Z,3,3
    en tenseur 5D X,Y,Z,1,6.
    """

    if tensor_matrix.ndim != 5:
        raise ValueError(
            f"Expected matrix tensor X,Y,Z,3,3, got {tensor_matrix.shape}"
        )

    if tensor_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected last dims 3,3, got {tensor_matrix.shape[-2:]}"
        )

    shape = tensor_matrix.shape[:3]
    out = np.zeros(shape + (1, 6), dtype=np.float32)

    out[..., 0, COMPONENT_TO_ANTS_INDEX["txx"]] = tensor_matrix[..., 0, 0]
    out[..., 0, COMPONENT_TO_ANTS_INDEX["txy"]] = tensor_matrix[..., 0, 1]
    out[..., 0, COMPONENT_TO_ANTS_INDEX["txz"]] = tensor_matrix[..., 0, 2]
    out[..., 0, COMPONENT_TO_ANTS_INDEX["tyy"]] = tensor_matrix[..., 1, 1]
    out[..., 0, COMPONENT_TO_ANTS_INDEX["tyz"]] = tensor_matrix[..., 1, 2]
    out[..., 0, COMPONENT_TO_ANTS_INDEX["tzz"]] = tensor_matrix[..., 2, 2]

    return out


# ============================================================
# Chargement / sauvegarde NIfTI
# ============================================================

def load_tensor_5d(path):
    path = Path(path)

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float64)

    tensor_matrix = tensor5d_to_matrix(data)

    return img, tensor_matrix


def load_all_tensors_5d(tensor_paths):
    """
    Charge tous les tenseurs 5D.

    Retourne :
    - ref_img
    - all_tensors : N,X,Y,Z,3,3
    """

    all_tensors = []
    ref_img = None
    ref_shape = None

    for path in tensor_paths:
        print(f"Loading: {path}")

        img, tensor = load_tensor_5d(path)

        if ref_img is None:
            ref_img = img
            ref_shape = tensor.shape[:3]
        else:
            if tensor.shape[:3] != ref_shape:
                raise ValueError(
                    f"Shape mismatch for {path}: "
                    f"{tensor.shape[:3]}, expected {ref_shape}"
                )

        all_tensors.append(tensor)

    all_tensors = np.asarray(all_tensors, dtype=np.float64)

    return ref_img, all_tensors


def save_tensor_5d(tensor_matrix, ref_img, out_path):
    """
    Sauvegarde un tenseur X,Y,Z,3,3 au format NIfTI 5D X,Y,Z,1,6.
    """

    out_path = Path(out_path)

    tensor5d = matrix_to_tensor5d(tensor_matrix)

    out_img = nib.Nifti1Image(
        tensor5d.astype(np.float32),
        ref_img.affine,
        ref_img.header.copy()
    )

    hdr = out_img.header
    hdr.set_data_dtype(np.float32)

    shape = tensor_matrix.shape[:3]

    # Layout explicite : X,Y,Z,1,6
    hdr["dim"][0] = 5
    hdr["dim"][1] = shape[0]
    hdr["dim"][2] = shape[1]
    hdr["dim"][3] = shape[2]
    hdr["dim"][4] = 1
    hdr["dim"][5] = 6
    hdr["dim"][6] = 1
    hdr["dim"][7] = 1

    # NIFTI_INTENT_SYMMATRIX = 1005, matrice 3x3
    hdr["intent_code"] = 1005
    hdr["intent_p1"] = 3

    nib.save(out_img, str(out_path))

    return out_path


# ============================================================
# Fusion 1 : moyenne Log-Euclidienne
# ============================================================

def compute_logeuclidean_mean(all_tensors):
    """
    all_tensors : N,X,Y,Z,3,3
    retourne : X,Y,Z,3,3
    """

    n_subjects = all_tensors.shape[0]
    shape = all_tensors.shape[1:4]

    mean_tensor = np.zeros(shape + (3, 3), dtype=np.float64)

    for index in np.ndindex(shape):
        logs = []

        for s in range(n_subjects):
            D = all_tensors[s, index[0], index[1], index[2]]
            D_spd = make_spd(D)
            logs.append(tensor_log(D_spd))

        mean_log = np.mean(logs, axis=0)

        D_mean = tensor_exp(mean_log)
        D_mean = 0.5 * (D_mean + D_mean.T)

        mean_tensor[index] = D_mean

    return mean_tensor


# ============================================================
# Fusion 2 : médiane eigen
# ============================================================

def compute_eigen_median(all_tensors, vertical_axis=VERTICAL_AXIS):
    """
    Fusion robuste par médiane des valeurs propres et des directions propres.

    all_tensors : N,X,Y,Z,3,3
    retourne : X,Y,Z,3,3
    """

    n_subjects = all_tensors.shape[0]
    shape = all_tensors.shape[1:4]

    median_tensor = np.zeros(shape + (3, 3), dtype=np.float64)

    vertical = normalize(vertical_axis)

    if np.linalg.norm(vertical) < 1e-12:
        raise ValueError("VERTICAL_AXIS has near-zero norm.")

    for index in np.ndindex(shape):
        eigenvalues = []
        eigenvectors = []

        for s in range(n_subjects):
            D = all_tensors[s, index[0], index[1], index[2]]
            D_spd = make_spd(D)

            vals, vecs = np.linalg.eigh(D_spd)

            # np.linalg.eigh sort en ordre croissant.
            # On inverse pour avoir lambda1 >= lambda2 >= lambda3.
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            # Tri selon l’alignement à l’axe vertical.
            inclinations = np.abs(vecs.T @ vertical)
            order_axis = np.argsort(inclinations)[::-1]

            vals = vals[order_axis]
            vecs = vecs[:, order_axis]

            # Correction du signe : v et -v sont équivalents.
            for k in range(3):
                if np.dot(vecs[:, k], vertical) < 0:
                    vecs[:, k] *= -1.0

            eigenvalues.append(vals)
            eigenvectors.append(vecs)

        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors)

        # Médiane des valeurs propres
        med_vals = np.median(eigenvalues, axis=0)
        med_vals = np.clip(med_vals, EPS, None)

        # Médiane composante par composante des vecteurs propres
        med_vecs = np.zeros((3, 3), dtype=np.float64)

        for k in range(3):
            v = np.median(eigenvectors[:, :, k], axis=0)
            v = normalize(v)

            # Fallback si la médiane donne un vecteur nul
            if np.linalg.norm(v) < 1e-12:
                v = eigenvectors[0, :, k]

            med_vecs[:, k] = v

        # Ré-orthogonalisation par SVD
        U, _, Vt = np.linalg.svd(med_vecs)
        med_vecs = U @ Vt

        # Correction pour obtenir une base directe
        if np.linalg.det(med_vecs) < 0:
            med_vecs[:, -1] *= -1.0

        D_median = med_vecs @ np.diag(med_vals) @ med_vecs.T
        D_median = 0.5 * (D_median + D_median.T)

        median_tensor[index] = D_median

    return median_tensor


# ============================================================
# Pipeline principal
# ============================================================

def find_tensor_files(tensor_dir, pattern):
    tensor_dir = Path(tensor_dir)

    if not tensor_dir.exists():
        raise FileNotFoundError(f"Tensor directory does not exist: {tensor_dir}")

    if not tensor_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {tensor_dir}")

    tensor_paths = sorted(tensor_dir.glob(pattern))

    if len(tensor_paths) == 0:
        raise FileNotFoundError(
            f"No tensor files found in {tensor_dir} with pattern '{pattern}'"
        )

    return tensor_paths


def build_tensor_atlases(tensor_dir, outdir, pattern):
    tensor_dir = Path(tensor_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tensor_paths = find_tensor_files(tensor_dir, pattern)

    print()
    print(f"Found {len(tensor_paths)} tensor files")
    print(f"Input directory: {tensor_dir.resolve()}")
    print(f"Output directory: {outdir.resolve()}")
    print()

    ref_img, all_tensors = load_all_tensors_5d(tensor_paths)

    print()
    print(f"Loaded tensors shape: {all_tensors.shape}")
    print()

    print("Computing Log-Euclidean mean atlas...")
    log_mean = compute_logeuclidean_mean(all_tensors)

    log_path = outdir / "atlas_logeuclidean_mean_tensor_5d.nii.gz"
    save_tensor_5d(log_mean, ref_img, log_path)

    print(f"Saved Log-Euclidean atlas: {log_path.resolve()}")
    print()

    print("Computing eigen median atlas...")
    eigen_median = compute_eigen_median(all_tensors)

    median_path = outdir / "atlas_eigen_median_tensor_5d.nii.gz"
    save_tensor_5d(eigen_median, ref_img, median_path)

    print(f"Saved eigen median atlas: {median_path.resolve()}")
    print()

    return log_path, median_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fuse 5D diffusion tensor NIfTI files into tensor atlases "
            "using Log-Euclidean mean and eigen median."
        )
    )

    parser.add_argument(
        "--tensor-dir",
        required=True,
        help="Directory containing input 5D tensor NIfTI files."
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for fused tensor atlases."
    )

    parser.add_argument(
        "--pattern",
        default="*_tensor_5d.nii.gz",
        help="Glob pattern used to find tensor files. Default: *_tensor_5d.nii.gz"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_tensor_atlases(
        tensor_dir=args.tensor_dir,
        outdir=args.outdir,
        pattern=args.pattern
    )


if __name__ == "__main__":
    main()