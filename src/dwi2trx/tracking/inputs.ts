/**
 * Assemble the WebGPU tracker's inputs from the app's loaded data: read the DWI
 * and FA NIfTIs into the exact memory layout the WGSL kernels index, and wire
 * the Stage 3b host matrices (sphere SH sampling, OPDT H/R/delta_b/delta_q,
 * b0s_mask) into a `TrackingInputs`. Also seeds from the FA mask.
 *
 * Layouts (from tracking_helpers.wgsl):
 *   dataf      x*dimy*dimz*dimt + y*dimz*dimt + z*dimt + t   (t fastest)
 *   metric_map x*dimy*dimz       + y*dimz       + z          (z fastest)
 * NIfTI memory is x-fastest, so both are reordered on load.
 */

import { gradientTable } from './gradients'
import { opdtMatrices } from './opdt'
import { realShDescoteaux } from './sh'
import { type Sphere, sphereThetaPhi } from './sphere'
import type { TrackingInputs } from './tracker'

// --- minimal NIfTI-1 image reader (gzip-aware, applies scl_slope/inter) ---

interface NiftiVol {
  nx: number
  ny: number
  nz: number
  nt: number
  /** Voxels in raw NIfTI order (x fastest), scaled to real values. */
  raw: Float32Array
  /** Voxel→world (RASMM) 4x4 row-major, from the sform (or pixdim fallback). */
  affine: number[][]
}

function readAffine(dv: DataView): number[][] {
  const sformCode = dv.getInt16(254, true)
  if (sformCode > 0) {
    const row = (o: number): number[] => [
      dv.getFloat32(o, true),
      dv.getFloat32(o + 4, true),
      dv.getFloat32(o + 8, true),
      dv.getFloat32(o + 12, true),
    ]
    return [row(280), row(296), row(312), [0, 0, 0, 1]] // srow_x / y / z
  }
  // No sform: fall back to a pixdim-scaled diagonal (origin at voxel 0).
  const px = (o: number) => dv.getFloat32(o, true) || 1
  return [
    [px(80), 0, 0, 0],
    [0, px(84), 0, 0],
    [0, 0, px(88), 0],
    [0, 0, 0, 1],
  ]
}

async function gunzipAll(file: File): Promise<Uint8Array> {
  const sig = new Uint8Array(await file.slice(0, 2).arrayBuffer())
  if (!(sig[0] === 0x1f && sig[1] === 0x8b)) {
    return new Uint8Array(await file.arrayBuffer())
  }
  const reader = file
    .stream()
    .pipeThrough(new DecompressionStream('gzip'))
    .getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    total += value.length
  }
  const out = new Uint8Array(total)
  let o = 0
  for (const c of chunks) {
    out.set(c, o)
    o += c.length
  }
  return out
}

function readNifti(bytes: Uint8Array): NiftiVol {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (dv.getInt32(0, true) !== 348) {
    throw new Error(
      'tracking: only NIfTI-1 (sizeof_hdr 348, little-endian) is supported.',
    )
  }
  const dim0 = dv.getInt16(40, true)
  const nx = dv.getInt16(42, true)
  const ny = dv.getInt16(44, true)
  const nz = dv.getInt16(46, true)
  const nt = dim0 >= 4 ? Math.max(1, dv.getInt16(48, true)) : 1
  const datatype = dv.getInt16(70, true)
  let slope = dv.getFloat32(112, true)
  const inter = dv.getFloat32(116, true)
  if (slope === 0) slope = 1
  const voxOffset = Math.round(dv.getFloat32(108, true)) || 352

  const n = nx * ny * nz * nt
  const raw = new Float32Array(n)
  const base = bytes.byteOffset + voxOffset
  const d = new DataView(bytes.buffer, base)
  // Datatype codes per the NIfTI-1 spec.
  const readers: Record<number, [number, (o: number) => number]> = {
    2: [1, (o) => d.getUint8(o)],
    4: [2, (o) => d.getInt16(o, true)],
    8: [4, (o) => d.getInt32(o, true)],
    16: [4, (o) => d.getFloat32(o, true)],
    64: [8, (o) => d.getFloat64(o, true)],
    256: [1, (o) => d.getInt8(o)],
    512: [2, (o) => d.getUint16(o, true)],
    768: [4, (o) => d.getUint32(o, true)],
  }
  const r = readers[datatype]
  if (!r) throw new Error(`tracking: unsupported NIfTI datatype ${datatype}.`)
  const [size, get] = r
  for (let i = 0; i < n; i++) raw[i] = get(i * size) * slope + inter
  return { nx, ny, nz, nt, raw, affine: readAffine(dv) }
}

// raw (x fastest) → dataf (x*ny*nz*nt + y*nz*nt + z*nt + t), t fastest.
function reorderDataf(v: NiftiVol): Float32Array {
  const { nx, ny, nz, nt, raw } = v
  const out = new Float32Array(raw.length)
  const sliceXY = nx * ny
  for (let t = 0; t < nt; t++) {
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const src = x + y * nx + z * sliceXY + t * sliceXY * nz
          const dst = ((x * ny + y) * nz + z) * nt + t
          out[dst] = raw[src]
        }
      }
    }
  }
  return out
}

// raw (x fastest) → metric_map (x*ny*nz + y*nz + z), z fastest.
function reorderMetric(v: NiftiVol): Float32Array {
  const { nx, ny, nz, raw } = v
  const out = new Float32Array(nx * ny * nz)
  const sliceXY = nx * ny
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        out[(x * ny + y) * nz + z] = raw[x + y * nx + z * sliceXY]
      }
    }
  }
  return out
}

// --- SH order: largest even order whose coeff count fits the direction count ---
// (nCoeff > nDir makes hat(B)=I so the residual
// bootstrap is a no-op; order 4 = 15 coeffs fits the 20-dir sample.)
const nCoeffForOrder = (order: number): number =>
  ((order + 1) * (order + 2)) / 2

export function chooseShOrder(nDir: number): number {
  if (nDir < nCoeffForOrder(2)) {
    throw new Error(
      `too few diffusion directions to fit an ODF (have ${nDir}, need ≥ ${nCoeffForOrder(2)}).`,
    )
  }
  let best = 2
  for (const o of [2, 4, 6, 8]) if (nCoeffForOrder(o) <= nDir) best = o
  return best
}

const f32 = (a: ArrayLike<number>): Float32Array => Float32Array.from(a)

/** Inputs for the GPU tracker plus the geometry needed to write a TRX. */
export interface AssembledInputs {
  inputs: TrackingInputs
  /** Voxel→world (RASMM) 4x4 row-major, for the TRX VOXEL_TO_RASMM. */
  voxelToRasmm: number[][]
  dims3: [number, number, number]
}

/** Build the full TrackingInputs for the Boot/OPDT tracker, plus TRX geometry. */
export async function assembleTrackingInputs(
  dwiFile: File,
  faFile: File,
  bvalText: string,
  bvecText: string,
  sphere: Sphere,
): Promise<AssembledInputs> {
  const gt = gradientTable(bvalText, bvecText)
  const nDir = gt.dwiTheta.length
  const shOrder = chooseShOrder(nDir)

  const opdt = opdtMatrices(gt.dwiTheta, gt.dwiPhi, shOrder)
  const { theta, phi } = sphereThetaPhi(sphere)
  const sampling = realShDescoteaux(theta, phi, shOrder).B // nVerts × nCoeff

  const dwi = readNifti(await gunzipAll(dwiFile))
  const fa = readNifti(await gunzipAll(faFile))
  if (fa.nx !== dwi.nx || fa.ny !== dwi.ny || fa.nz !== dwi.nz) {
    throw new Error(
      'tracking: FA map and DWI have different spatial dimensions.',
    )
  }
  if (dwi.nt !== gt.b0sMask.length) {
    throw new Error(
      `tracking: DWI has ${dwi.nt} volumes but bval lists ${gt.b0sMask.length}.`,
    )
  }

  return {
    inputs: {
      dataf: reorderDataf(dwi),
      dims: [dwi.nx, dwi.ny, dwi.nz, dwi.nt],
      metricMap: reorderMetric(fa),
      sphereVertices: sphere.vertices,
      sphereEdges: sphere.edges,
      H: f32(opdt.H),
      R: f32(opdt.R),
      deltaB: f32(opdt.deltaB),
      deltaQ: f32(opdt.deltaQ),
      samplingMatrix: f32(sampling),
      b0sMask: Int32Array.from(gt.b0sMask, (b) => (b ? 1 : 0)),
      samplmNr: sphere.nVerts,
      nedges: sphere.nEdges,
      deltaNr: opdt.nCoeff,
      modelType: 0, // OPDT
    },
    voxelToRasmm: dwi.affine,
    dims3: [dwi.nx, dwi.ny, dwi.nz],
  }
}

/**
 * Seeds (VOX coords, flat x,y,z) from voxels whose metric (FA) ≥ `threshold`.
 * `perAxis`³ seeds per voxel on a regular sub-grid (deterministic — no RNG to
 * match), capped at `maxSeeds`. metric_map is in (x,y,z) z-fastest order.
 */
export function seedsFromMask(
  metricMap: Float32Array,
  dims: [number, number, number, number],
  threshold: number,
  perAxis = 1,
  maxSeeds = 100000,
): Float32Array {
  const [nx, ny, nz] = dims
  const offs: number[] = []
  for (let i = 0; i < perAxis; i++) offs.push((i + 0.5) / perAxis - 0.5) // voxel-centred sub-grid
  const seeds: number[] = []
  outer: for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        if (metricMap[(x * ny + y) * nz + z] < threshold) continue
        for (const dx of offs)
          for (const dy of offs)
            for (const dz of offs) {
              seeds.push(x + dx, y + dy, z + dz)
              if (seeds.length / 3 >= maxSeeds) break outer
            }
      }
    }
  }
  return Float32Array.from(seeds)
}
