# spinal-atlas

This folder contains small scripts to build and use diffusion tensor images in a common reference space.

The general workflow is:

1. Extract a b0 image from each corrected DWI.
2. Convert the six tensor component images into one 5D tensor NIfTI.
3. Register each subject b0 image to the reference subject b0.
4. Apply the existing b0 transforms to the subject tensor image.
5. Optionally convert tensors to `.fib.gz` for DSI Studio.
6. Fuse registered tensors into tensor atlases.

## Files

### `extract_b0.py`

Extracts a b0 image from a 4D DWI using the `.bval` file.

Example:

```bash
python extract_b0.py \
  --dwi sub-27_Corrected.nii.gz \
  --bval sub-27_Corrected.bval \
  --bvec sub-27_Corrected.bvec \
  --out sub-27_b0.nii.gz
```

By default, all volumes with `bval == 0` are averaged.

---

### `convert_components_to_tensor5d.py`

Combines six tensor component images into one 5D tensor NIfTI.

Expected components:

```text
txx, txy, txz, tyy, tyz, tzz
```

The output shape is:

```text
X, Y, Z, 1, 6
```

Example:

```bash
python convert_components_to_tensor5d.py \
  --subject sub-27 \
  --tensor-root sub-27_tensor_components \
  --component-pattern "{subject}_Corrected_{component}.nii.gz" \
  --out sub-27_tensor_components/sub-27_tensor_5d.nii.gz
```

---

### `register_b0_only.py`

Registers a moving subject b0 image to a fixed/reference b0 image using ANTs SyN.

Example:

```bash
python register_b0_only.py \
  --fixed-b0 ../sub-28/sub-28_b0.nii.gz \
  --moving-b0 sub-27_b0.nii.gz \
  --outdir b0_registration_sub-27_to_sub-28 \
  --prefix sub-27_to_sub-28_
```

This creates the ANTs transform files:

```text
0GenericAffine.mat
1Warp.nii.gz
1InverseWarp.nii.gz
Warped.nii.gz
```

It also creates simple QC outputs comparing the fixed b0 and warped b0.

---

### `apply_existing_transforms_to_tensor.py`

Applies the transforms computed from the b0 registration to a 5D tensor image.

Example:

```bash
python apply_existing_transforms_to_tensor.py \
  --tensor sub-27_tensor_components/sub-27_tensor_5d.nii.gz \
  --reference ../sub-28/sub-28_b0.nii.gz \
  --warp b0_registration_sub-27_to_sub-28/sub-27_to_sub-28_1Warp.nii.gz \
  --affine b0_registration_sub-27_to_sub-28/sub-27_to_sub-28_0GenericAffine.mat \
  --out ../atlas/sub-27_tensor_to_sub-28_reoriented.nii.gz \
  --split \
  --split-prefix sub-27_to_sub-28
```

The transform direction is assumed to be:

```text
moving subject space -> reference subject space
```

---

### `convert_tensor_to_fib.py`

Converts a 5D tensor NIfTI into a `.fib.gz` file for DSI Studio.

It uses an existing `.fib.gz` file (the reference used for the registration) as a template, mainly to reuse fields such as `odf_vertices`.

Example:

```bash
python convert_tensor_to_fib.py \
  --tensor sub-27_tensor_components/sub-27_tensor_5d.nii.gz \
  --template-fib ../sub-28/sub-28_Corrected.fib.gz \
  --out sub-27_tensor_components/sub-27_tensor_5d.fib.gz \
  --index-base zero
```

The same script can also be used on reoriented tensors or atlas tensors.

---

### `fuse_tensors.py`

Fuses several registered 5D tensor images into atlas tensors.

It creates two outputs:

```text
atlas_logeuclidean_mean_tensor_5d.nii.gz
atlas_eigen_median_tensor_5d.nii.gz
```

Example:

```bash
python fuse_tensors.py \
  --tensor-dir atlas \
  --outdir atlas \
  --pattern "*_tensor_to_sub-28_reoriented.nii.gz"
```

The Log-Euclidean mean is the standard average in tensor log-space.
The eigen median is a more robust fusion based on eigenvalues and eigenvectors.

---

### `script.py`

Main driver script showing how to run the full pipeline for several subjects.

It defines:

```python
ROOT = Path("/Users/frindel/Documents/Données-Corentin")
REFERENCE_SUBJECT = "sub-28"
SUBJECTS = ["sub-27", "sub-29"]
```

Most steps are currently commented out, so you can uncomment only the steps you want to run.

## Typical order of use

For each subject:

```text
extract_b0.py
convert_components_to_tensor5d.py
register_b0_only.py
apply_existing_transforms_to_tensor.py
convert_tensor_to_fib.py   optional
```

After all subjects have been registered to the reference space:

```text
fuse_tensors.py
convert_tensor_to_fib.py   optional, for atlas outputs
```

## Important assumptions

The scripts assume tensor components are stored in ANTs/ITK order:

```text
[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
```

The 5D tensor format is expected to be:

```text
X, Y, Z, 1, 6
```

All tensor component images for one subject must have the same shape and affine.

ANTs must be installed and available in the environment, or provided with `--ants-bin`.
