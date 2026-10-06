#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Register one b0 image to another using ANTs SyN.

This script does ONLY the b0 registration step:
    moving b0 -> fixed/reference b0

It creates:
    - ANTs transform files
    - warped moving b0 in fixed space
    - fixed-minus-warped difference image
    - absolute difference image
    - a small text QC report with correlation / MAE / RMSE
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def run(cmd, dry_run=False):
    print("\nCOMMAND:")
    print(" ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], check=True)


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n"
            f"  raw: {path}\n"
            f"  abs: {path.resolve()}\n"
            f"  cwd: {Path.cwd()}"
        )
    return path


def check_ants_available(ants_bin=None, explicit=False):
    if ants_bin:
        os.environ["PATH"] = str(Path(ants_bin)) + os.pathsep + os.environ["PATH"]

    registration_program = "antsRegistration" if explicit else "antsRegistrationSyN.sh"
    ants_reg = shutil.which(registration_program)
    ants_apply = shutil.which("antsApplyTransforms")

    if ants_reg is None:
        raise RuntimeError(
            f"{registration_program} not found. "
            "Use --ants-bin /path/to/ants/bin"
        )
    if ants_apply is None:
        raise RuntimeError(
            "antsApplyTransforms not found. "
            "Use --ants-bin /path/to/ants/bin"
        )

    print(f"ANTs registration: {ants_reg}")
    print(f"ANTs apply:        {ants_apply}")


def save_like(data, ref_img, out_path):
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    out.set_qform(ref_img.get_qform(), code=int(ref_img.header["qform_code"]))
    out.set_sform(ref_img.get_sform(), code=int(ref_img.header["sform_code"]))
    out.header.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


def make_qc_outputs(fixed_b0, warped_b0, outdir, prefix):
    fixed_img = nib.load(str(fixed_b0))
    warped_img = nib.load(str(warped_b0))

    fixed = fixed_img.get_fdata(dtype=np.float32)
    warped = warped_img.get_fdata(dtype=np.float32)

    if fixed.shape != warped.shape:
        raise ValueError(
            f"Shape mismatch for QC: fixed {fixed.shape}, warped {warped.shape}"
        )

    diff = fixed - warped
    absdiff = np.abs(diff)

    diff_path = outdir / f"{prefix}b0_diff_fixed_minus_warped.nii.gz"
    absdiff_path = outdir / f"{prefix}b0_absdiff.nii.gz"
    report_path = outdir / f"{prefix}b0_registration_QC.txt"

    save_like(diff, fixed_img, diff_path)
    save_like(absdiff, fixed_img, absdiff_path)

    # Avoid background dominating too much: report both full-image and nonzero-mask metrics.
    fixed_flat = fixed.ravel()
    warped_flat = warped.ravel()

    corr_full = np.corrcoef(fixed_flat, warped_flat)[0, 1]
    mae_full = float(np.mean(absdiff))
    rmse_full = float(np.sqrt(np.mean(diff ** 2)))

    mask = (fixed > 0) | (warped > 0)
    if np.any(mask):
        corr_mask = np.corrcoef(fixed[mask].ravel(), warped[mask].ravel())[0, 1]
        mae_mask = float(np.mean(absdiff[mask]))
        rmse_mask = float(np.sqrt(np.mean(diff[mask] ** 2)))
        n_mask = int(mask.sum())
    else:
        corr_mask = np.nan
        mae_mask = np.nan
        rmse_mask = np.nan
        n_mask = 0

    text = f"""B0 registration QC
==================

Fixed b0:
  {fixed_b0}

Warped moving b0:
  {warped_b0}

Outputs:
  Difference fixed - warped:
    {diff_path}
  Absolute difference:
    {absdiff_path}

Shapes:
  fixed:  {fixed.shape}
  warped: {warped.shape}

Full-image metrics:
  correlation: {corr_full:.6f}
  MAE:         {mae_full:.6f}
  RMSE:        {rmse_full:.6f}

Nonzero-mask metrics, mask = fixed > 0 OR warped > 0:
  n voxels:    {n_mask}
  correlation: {corr_mask:.6f}
  MAE:         {mae_mask:.6f}
  RMSE:        {rmse_mask:.6f}
"""

    report_path.write_text(text, encoding="utf-8")

    print("\n=== QC outputs ===")
    print(f"Difference:     {diff_path}")
    print(f"Abs difference: {absdiff_path}")
    print(f"Report:         {report_path}")
    print(f"Correlation full: {corr_full:.4f}")
    print(f"Correlation mask: {corr_mask:.4f}")
    print("==================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register a moving b0 image to a fixed/reference b0 image with ANTs SyN."
    )
    parser.add_argument("--fixed-b0", required=True, help="Reference b0, e.g. sub-27_b0.nii.gz")
    parser.add_argument("--moving-b0", required=True, help="Moving b0, e.g. sub-28_b0.nii.gz")
    parser.add_argument("--outdir", default="b0_registration", help="Output directory")
    parser.add_argument("--prefix", default="moving_to_fixed_", help="Output filename prefix")
    parser.add_argument(
        "--ants-bin",
        default="/opt/miniconda3/envs/ants_tensor/bin",
        help="Directory containing antsRegistrationSyN.sh and antsApplyTransforms",
    )
    parser.add_argument(
        "--transform-type",
        default="s",
        choices=["t", "r", "a", "s", "sr", "so", "b", "br", "bo"],
        help="ANTs transform type. Default s = rigid + affine + SyN",
    )
    parser.add_argument("--grad-step", type=float, help="Explicit SyN gradient step")
    parser.add_argument("--reg-iterations", help="SyN iterations per level, e.g. 5000x5000x50")
    parser.add_argument("--aff-iterations", help="Affine iterations per level, e.g. 2000x500x500x10")
    parser.add_argument("--aff-sampling", type=int, help="Affine Mattes MI histogram bins")
    parser.add_argument("--random-seed", type=int, help="Explicit registration seed")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    explicit = (args.grad_step, args.reg_iterations, args.aff_iterations, args.aff_sampling)
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit) or args.random_seed is None:
            parser.error("Provide all four explicit parameters and --random-seed together")
        if args.transform_type != "s":
            parser.error("Explicit parameters currently support --transform-type s only")
        if not np.isfinite(args.grad_step) or args.grad_step <= 0 or args.aff_sampling < 2:
            parser.error("grad-step must be positive and aff-sampling at least 2")
        # The recovered search used three nonlinear and four linear levels.
        for value, levels in ((args.reg_iterations, 3), (args.aff_iterations, 4)):
            parts = value.split("x")
            if len(parts) != levels or not all(part.isdigit() for part in parts) or not any(int(part) for part in parts):
                parser.error(f"Expected {levels} nonnegative iteration counts separated by x")
    if args.threads < 1 or (args.random_seed is not None and args.random_seed < 1):
        parser.error("threads and random-seed must be positive")
    return args


def main():
    args = parse_args()

    print("\n=== B0 REGISTRATION ONLY ===")
    print("cwd:", Path.cwd())
    print("fixed_b0 raw:", args.fixed_b0)
    print("moving_b0 raw:", args.moving_b0)
    print("outdir raw:", args.outdir)
    print("prefix:", args.prefix)

    check_ants_available(args.ants_bin, explicit=args.grad_step is not None)

    fixed_b0 = require_file(args.fixed_b0, "fixed/reference b0")
    moving_b0 = require_file(args.moving_b0, "moving b0")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_prefix = outdir / args.prefix

    # Default behavior: keep the original ANTs shell wrapper.
    if args.grad_step is None:
        cmd = [
            "antsRegistrationSyN.sh", "-d", "3",
            "-f", fixed_b0, "-m", moving_b0, "-o", output_prefix,
            "-t", args.transform_type, "-n", str(args.threads),
        ]
        if args.random_seed is not None:
            cmd += ["-e", str(args.random_seed)]
    else:
        # New explicit b0 registration, not a proven historical optimum.
        # Per-level settings follow the archived three-/four-level search;
        # affine + SyN settings here are explicitly applied rather than inherited.
        for path in (fixed_b0, moving_b0, output_prefix):
            if any(c in str(path) for c in ",[]"):
                raise ValueError("ANTs paths must not contain commas or brackets")
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
        cmd = [
            "antsRegistration", "--dimensionality", "3",
            "--random-seed", str(args.random_seed), "--verbose", "1",
            "--interpolation", "Linear", "--collapse-output-transforms", "1",
            "--write-composite-transform", "0",
            "--output", f"[{output_prefix},{output_prefix}Warped.nii.gz]",
            "--initial-moving-transform", f"[{fixed_b0},{moving_b0},1]",
            "--transform", "Affine[0.25]",
            "--metric", f"Mattes[{fixed_b0},{moving_b0},1,{args.aff_sampling},Regular,0.2]",
            "--convergence", f"[{args.aff_iterations},1e-6,10]",
            "--shrink-factors", "4x2x2x1", "--smoothing-sigmas", "3x2x1x0vox",
            "--transform", f"SyN[{args.grad_step},3,0]",
            "--metric", f"Mattes[{fixed_b0},{moving_b0},1,32]",
            "--convergence", f"[{args.reg_iterations},1e-7,8]",
            "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
        ]
    run(cmd, dry_run=args.dry_run)

    warped_b0 = Path(str(output_prefix) + "Warped.nii.gz")
    affine = Path(str(output_prefix) + "0GenericAffine.mat")
    warp = Path(str(output_prefix) + "1Warp.nii.gz")
    inverse_warp = Path(str(output_prefix) + "1InverseWarp.nii.gz")

    print("\nExpected outputs:")
    print("  warped b0:   ", warped_b0)
    print("  affine:      ", affine)
    print("  warp:        ", warp)
    print("  inverse warp:", inverse_warp)

    if not args.dry_run:
        require_file(warped_b0, "warped moving b0")
        require_file(affine, "affine transform")
        require_file(warp, "SyN warp")
        make_qc_outputs(fixed_b0, warped_b0, outdir, args.prefix)

    print("Done.")


if __name__ == "__main__":
    main()
