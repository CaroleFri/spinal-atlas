#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import shutil
import argparse
import tempfile

import numpy as np
import nibabel as nib
from scipy.io import loadmat, savemat


# Ordre ANTs/ITK si ton tenseur 5D vient du script précédent :
# [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
COMPONENT_TO_INDEX = {
    "txx": 0,
    "txy": 1,
    "tyy": 2,
    "txz": 3,
    "tyz": 4,
    "tzz": 5,
}


def load_fib_mat(fib_gz_path):
    """
    Charge un .fib.gz DSI Studio comme fichier MATLAB v4.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        fib_path = os.path.join(tmpdir, "tmp.fib")
        print(fib_gz_path)
        with gzip.open(fib_gz_path, "rb") as f_in:
            with open(fib_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        fib = loadmat(fib_path)

    return fib


def save_fib_gz(mat_data, out_fib_gz):
    """
    Sauvegarde un .fib.gz MATLAB v4.
    """

    if not out_fib_gz.endswith(".fib.gz"):
        raise ValueError("La sortie doit se terminer par .fib.gz")

    out_fib = out_fib_gz[:-3]

    savemat(out_fib, mat_data, format="4")

    with open(out_fib, "rb") as f_in:
        with gzip.open(out_fib_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(out_fib)


def tensor5d_to_fib_with_index0(
    tensor_path,
    base_fib_gz,
    out_fib_gz,
    eps=1e-8,
    index_base="zero",
    verbose=True,
):
    """
    Convertit un tenseur 5D NIfTI en .fib.gz compatible DSI Studio
    avec fa0 + index0 + odf_vertices.

    index_base:
        "zero" -> index0 entre 0 et N-1
        "one"  -> index0 entre 1 et N
    """

    img = nib.load(tensor_path)
    tensor5d = img.get_fdata(dtype=np.float64)

    if tensor5d.ndim != 5:
        raise ValueError(f"Tenseur attendu en 5D, shape obtenue: {tensor5d.shape}")

    if tensor5d.shape[3] != 1 or tensor5d.shape[4] != 6:
        raise ValueError(
            f"Shape attendue X,Y,Z,1,6 ; shape obtenue: {tensor5d.shape}"
        )

    nx, ny, nz = tensor5d.shape[:3]
    dim = (nx, ny, nz)

    voxel_size = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    dimension = np.array([nx, ny, nz], dtype=np.int32)

    if verbose:
        print(f"Loaded tensor: {tensor_path}")
        print(f"Tensor shape: {tensor5d.shape}")
        print(f"Dimension: {dimension}")
        print(f"Voxel size: {voxel_size}")

    # Récupère odf_vertices depuis un vrai .fib.gz DSI Studio
    base_fib = load_fib_mat(base_fib_gz)

    if "odf_vertices" not in base_fib:
        raise ValueError("Le fichier base_fib_gz ne contient pas odf_vertices")

    odf_vertices = np.asarray(base_fib["odf_vertices"], dtype=np.float32)

    # Selon les fichiers, odf_vertices peut être 3 x N ou N x 3.
    if odf_vertices.shape[0] == 3:
        vertices = odf_vertices.T.astype(np.float64)  # N x 3
        odf_vertices_out = odf_vertices
    elif odf_vertices.shape[1] == 3:
        vertices = odf_vertices.astype(np.float64)    # N x 3
        odf_vertices_out = odf_vertices.T
    else:
        raise ValueError(f"Shape inattendue pour odf_vertices: {odf_vertices.shape}")

    # Normalisation des vertices
    vertices_norm = np.linalg.norm(vertices, axis=1)
    valid = vertices_norm > eps
    vertices[valid] /= vertices_norm[valid, None]

    n_vertices = vertices.shape[0]

    if verbose:
        print(f"Loaded odf_vertices from: {base_fib_gz}")
        print(f"Number of ODF vertices: {n_vertices}")

    t = tensor5d[..., 0, :]

    D = np.zeros(dim + (3, 3), dtype=np.float64)

    D[..., 0, 0] = t[..., COMPONENT_TO_INDEX["txx"]]

    D[..., 0, 1] = t[..., COMPONENT_TO_INDEX["txy"]]
    D[..., 1, 0] = t[..., COMPONENT_TO_INDEX["txy"]]

    D[..., 0, 2] = t[..., COMPONENT_TO_INDEX["txz"]]
    D[..., 2, 0] = t[..., COMPONENT_TO_INDEX["txz"]]

    D[..., 1, 1] = t[..., COMPONENT_TO_INDEX["tyy"]]

    D[..., 1, 2] = t[..., COMPONENT_TO_INDEX["tyz"]]
    D[..., 2, 1] = t[..., COMPONENT_TO_INDEX["tyz"]]

    D[..., 2, 2] = t[..., COMPONENT_TO_INDEX["tzz"]]

    fa0_3d = np.zeros(dim, dtype=np.float32)
    index0_3d = np.zeros(dim, dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                tensor = D[i, j, k]
                tensor = 0.5 * (tensor + tensor.T)

                vals, vecs = np.linalg.eigh(tensor)

                order = np.argsort(vals)[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                vals = np.clip(vals, 0, None)

                l1, l2, l3 = vals
                md = (l1 + l2 + l3) / 3.0
                denom = l1**2 + l2**2 + l3**2

                if denom > eps:
                    fa = np.sqrt(
                        1.5
                        * (
                            (l1 - md) ** 2
                            + (l2 - md) ** 2
                            + (l3 - md) ** 2
                        )
                        / denom
                    )
                else:
                    fa = 0.0

                fa0_3d[i, j, k] = fa

                if fa <= 0:
                    index0_3d[i, j, k] = 0
                    continue

                v1 = vecs[:, 0]
                norm = np.linalg.norm(v1)

                if norm <= eps:
                    index0_3d[i, j, k] = 0
                    continue

                v1 = v1 / norm

                # Direction axiale : v et -v sont équivalents.
                # On prend le vertex qui maximise |dot(v, vertex)|.
                dots = np.abs(vertices @ v1)
                idx = int(np.argmax(dots))

                if index_base == "zero":
                    index0_3d[i, j, k] = idx
                elif index_base == "one":
                    index0_3d[i, j, k] = idx + 1
                else:
                    raise ValueError("index_base doit être 'zero' ou 'one'")

    fa0 = np.reshape(
        fa0_3d,
        (nx * ny, nz),
        order="F",
    ).astype(np.float32)

    index0 = np.reshape(
        index0_3d,
        (nx * ny, nz),
        order="F",
    ).astype(np.float32)

    # =========================
    # BUILD OUTPUT FIB FROM TEMPLATE
    # =========================

    mat_data = {}

    # On garde tous les champs du .fib initial/template
    for key, value in base_fib.items():
        if key.startswith("__"):
            continue
        mat_data[key] = value

    # Puis on remplace les champs calculés à partir du tenseur
    mat_data["dimension"] = dimension
    mat_data["voxel_size"] = voxel_size
    mat_data["fa0"] = fa0
    mat_data["index0"] = index0

    # On force odf_vertices dans le format attendu
    mat_data["odf_vertices"] = odf_vertices_out.astype(np.float32)

    # Si odf_faces existe dans le template, on le garde.
    # Sinon, il sera absent.
    if "odf_faces" in base_fib:
        mat_data["odf_faces"] = base_fib["odf_faces"]

    save_fib_gz(mat_data, out_fib_gz)

    if verbose:
        print(f"Saved: {out_fib_gz}")
        print(f"fa0 nonzero: {np.count_nonzero(fa0)}")
        print(f"index0 unique count: {len(np.unique(index0))}")
        print(f"index0 min/max: {np.min(index0)} / {np.max(index0)}")

    return out_fib_gz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert tensor 5D NIfTI to DSI Studio .fib.gz using fa0/index0/odf_vertices."
    )

    parser.add_argument(
        "--tensor",
        required=True,
        help="Tenseur 5D .nii.gz, shape X,Y,Z,1,6",
    )

    parser.add_argument(
        "--template-fib",
        required=True,
        help="Fichier .fib.gz initial utilisé comme template DSI Studio",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Sortie .fib.gz",
    )

    parser.add_argument(
        "--index-base",
        default="zero",
        choices=["zero", "one"],
        help="Convention d'indexation pour index0. Essayer zero d'abord.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tensor5d_to_fib_with_index0(
        tensor_path=args.tensor,
        base_fib_gz=args.template_fib,
        out_fib_gz=args.out,
        index_base=args.index_base,
    )


if __name__ == "__main__":
    main()