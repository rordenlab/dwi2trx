/**
 * Single source of truth for wizard + pipeline state — flat, no classes, no
 * setters.
 */

export type Step = 1 | 2 | 3

export type InputSource = 'sample' | 'nifti' | 'dicom'

/** A validated diffusion input: a NIfTI + matching bval/bvec (+ optional json). */
export interface DwiInput {
  nifti: File
  bval: File
  bvec: File
  json?: File
  /** Gradient directions = bval entries = NIfTI 4D volume count (all cross-checked). */
  directions: number
  source: InputSource
}

/** Tensor-fit outputs (Stage 2): FA (3D) + V1 (4D 3-vector), as `.nii.gz` Files. */
export interface TensorMaps {
  fa: File
  v1: File
}

export const state: {
  /** Active tab (1 select · 2 maps · 3 streamlines). */
  step: Step
  input?: DwiInput
  maps?: TensorMaps
  /** Stage 3c output: tracked streamlines in VOX space (flat x,y,z per line). */
  streamlines?: Float32Array[]
  tracts?: File // Stage 3d (TRX) — written from `streamlines`
} = { step: 1 }
