/**
 * Browser WebGPU Boot/OPDT streamline tracker — a port of GPUStreamlines'
 * `cuslines/webgpu/` host code (wg_tractography.py + wg_propagate_seeds.py +
 * wg_direction_getters.py) to a browser `GPUDevice`.
 *
 * Two-pass, chunked over seeds (mirrors WebGPUSeedBatchPropagator):
 *   pass 1 `getNumStreamlinesBoot_k` → per-seed streamline counts
 *   → readback + exclusive prefix sum (CPU) → write back offsets
 *   pass 2 `genStreamlinesMergeBoot_k` → streamline coordinates + lengths
 *   → readback → assemble per-streamline point arrays.
 *
 * BROWSER-ONLY and not headlessly verifiable: requires WebGPU with the
 * `subgroups` feature and `maxStorageBuffersPerShaderStage >= 10` (the Boot
 * kernel binds 10 storage buffers across 3 groups after the repack). The host math feeding this
 * (sphere, gradients, OPDT H/R/delta_b/delta_q/sampling_matrix/b0s_mask) is
 * golden-tested in tracking.test.ts; this orchestration is verified in-browser.
 */

import { BLOCK_Y, divUp, MAX_SLINE_LEN, REAL_SIZE } from './globals'
import { BOOT_ENTRY_GEN, BOOT_ENTRY_GETNUM, bootShaderSource } from './shaders'

/** Static per-dataset inputs uploaded once. `dataf` is the DWI signal in
 *  (x,y,z,t) C-contiguous order (t fastest); the OPDT matrices come straight
 *  from the Stage 3b host port (opdt.ts / sh.ts / gradients.ts). */
export interface TrackingInputs {
  dataf: Float32Array // dimx*dimy*dimz*dimt
  dims: [number, number, number, number] // dimx, dimy, dimz, dimt
  metricMap: Float32Array // stop map (FA), dimx*dimy*dimz
  sphereVertices: Float32Array // nverts*3
  sphereEdges: Int32Array // nedges*2
  // Boot/OPDT direction-getter matrices:
  H: Float32Array // (dimt-nb0) x (dimt-nb0)
  R: Float32Array // same
  deltaB: Float32Array // delta_nr x nCoeff
  deltaQ: Float32Array // delta_nr x nCoeff
  samplingMatrix: Float32Array // nverts x nCoeff
  b0sMask: Int32Array // dimt (1 where b0)
  samplmNr: number // sampling_matrix rows = nverts
  nedges: number // sphere_edges rows
  deltaNr: number // delta_b rows
  modelType: number // 0 = OPDT, 1 = CSA
}

export interface TrackingParams {
  maxAngle: number // radians (default radians(60))
  stepSize: number // voxels (default 0.5)
  tcThreshold: number // stop_threshold on metricMap (e.g. FA floor)
  relativePeakThresh: number // default 0.25
  minSeparationAngle: number // radians (default radians(45))
  minSignal: number // default 1
  rngSeed: number
  chunkSize: number // seeds per GPU batch (default 25000)
  minPts: number // discard streamlines shorter than this
  maxPts: number // discard streamlines longer than this
}

export const DEFAULT_PARAMS: Omit<TrackingParams, 'tcThreshold'> = {
  maxAngle: (60 * Math.PI) / 180,
  stepSize: 0.5,
  relativePeakThresh: 0.25,
  minSeparationAngle: (45 * Math.PI) / 180,
  minSignal: 1,
  rngSeed: 0,
  chunkSize: 25000,
  minPts: 0,
  maxPts: Number.POSITIVE_INFINITY,
}

/** The repacked Boot kernel binds 10 storage buffers across 3 groups (down
 *  from the reference's 17 — Dawn/Metal caps this at 10 in the browser). */
const REQUIRED_STORAGE_BUFFERS = 10

/** The Boot kernel's `var<workgroup>` arrays total ~20.8 KB; Dawn defaults the
 *  limit to the 16 KB spec minimum (wgpu-native defaults higher). */
const REQUIRED_WORKGROUP_STORAGE = 20800

/**
 * Request a WebGPU device able to run the Boot tracker. Throws a *specific*
 * diagnostic (and logs the adapter's features + key limits to the console) so
 * we can tell apart the distinct failure causes — no WebGPU, missing
 * `subgroups`, or too few storage buffers / workgroup storage per stage
 * (Dawn/Metal may cap below what the repacked Boot kernel needs).
 */
export async function getTrackingDevice(): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is not available in this browser.')
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  })
  if (!adapter) throw new Error('No WebGPU adapter found.')

  const features = [...adapter.features]
  const maxSB = adapter.limits.maxStorageBuffersPerShaderStage
  // Ground truth for diagnosing capability gaps in the user's browser.
  console.info(
    '[dwi2trx] WebGPU adapter — features:',
    features,
    '| maxStorageBuffersPerShaderStage:',
    maxSB,
    '| maxBufferSize:',
    adapter.limits.maxBufferSize,
  )

  // The browser spec feature is "subgroups" (wgpu-native calls it "subgroup").
  const hasSubgroups = adapter.features.has('subgroups')
  if (!hasSubgroups) {
    throw new Error(
      `your GPU/browser doesn't expose the WebGPU "subgroups" feature ` +
        `(adapter features: ${features.join(', ') || 'none'}). On Chrome try ` +
        `enabling chrome://flags/#enable-unsafe-webgpu; Safari/Firefox may not ` +
        `support subgroups yet.`,
    )
  }
  if (maxSB < REQUIRED_STORAGE_BUFFERS) {
    throw new Error(
      `your GPU exposes only ${maxSB} storage buffers per shader stage, but the ` +
        `tracker's Boot kernel needs ${REQUIRED_STORAGE_BUFFERS}.`,
    )
  }
  if (
    adapter.limits.maxComputeWorkgroupStorageSize < REQUIRED_WORKGROUP_STORAGE
  ) {
    throw new Error(
      `your GPU allows only ${adapter.limits.maxComputeWorkgroupStorageSize} bytes of ` +
        `compute workgroup storage, but the Boot kernel needs ${REQUIRED_WORKGROUP_STORAGE}.`,
    )
  }
  try {
    return await adapter.requestDevice({
      requiredFeatures: ['subgroups'] as GPUFeatureName[],
      requiredLimits: {
        maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS,
        maxBindGroups: 4,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        // Boot uses ~20.8 KB of workgroup storage; Dawn defaults the limit to
        // 16 KB (wgpu-native defaults higher, so the reference never set it).
        maxComputeWorkgroupStorageSize:
          adapter.limits.maxComputeWorkgroupStorageSize,
      },
    })
  } catch (err) {
    throw new Error(`WebGPU device request failed: ${(err as Error).message}`)
  }
}

// --- buffer helpers (browser has no createBufferWithData) ---

function storageFromData(
  device: GPUDevice,
  data: Float32Array | Int32Array,
  label: string,
): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
    label,
  })
  const Ctor = data instanceof Float32Array ? Float32Array : Int32Array
  new Ctor(buf.getMappedRange()).set(data as never)
  buf.unmap()
  return buf
}

function emptyStorage(
  device: GPUDevice,
  sizeBytes: number,
  label: string,
): GPUBuffer {
  return device.createBuffer({
    size: Math.max(sizeBytes, 4),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    label,
  })
}

/** Copy a storage buffer's first `byteLength` bytes back to the CPU. */
async function readback(
  device: GPUDevice,
  buf: GPUBuffer,
  byteLength: number,
): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(buf, 0, staging, 0, byteLength)
  device.queue.submit([enc.finish()])
  try {
    await staging.mapAsync(GPUMapMode.READ)
    return staging.getMappedRange().slice(0)
  } finally {
    staging.destroy() // also frees it if mapAsync rejects (device loss)
  }
}

/** Offsets (in f32 elements) of each matrix within the packed modelData buffer
 *  [H | R | delta_b | delta_q | sampling_matrix | b0s_mask]. off_H is 0. */
interface ModelOffsets {
  offR: number
  offDeltaB: number
  offDeltaQ: number
  offSampling: number
  offB0s: number
}

function modelOffsets(inputs: TrackingInputs): ModelOffsets {
  const offR = inputs.H.length
  const offDeltaB = offR + inputs.R.length
  const offDeltaQ = offDeltaB + inputs.deltaB.length
  const offSampling = offDeltaQ + inputs.deltaQ.length
  const offB0s = offSampling + inputs.samplingMatrix.length
  return { offR, offDeltaB, offDeltaQ, offSampling, offB0s }
}

// --- params struct (must match BootTrackingParams in boot.wgsl) ---
// 6 f32 + 17 i32 = 92 bytes, little-endian. The trailing 5 ints are the
// modelData offsets (the repack packs the 6 model matrices into one buffer).
// tc_threshold/step_size/rng_offset are zero on the getNum pass.
function packBootParams(
  inputs: TrackingInputs,
  off: ModelOffsets,
  p: TrackingParams,
  nseeds: number,
  rngOffset: number,
  forGen: boolean,
): ArrayBuffer {
  const buf = new ArrayBuffer(92)
  const dv = new DataView(buf)
  const seedLo = p.rngSeed & 0xffffffff
  const seedHi = Math.floor(p.rngSeed / 0x100000000) & 0xffffffff
  let o = 0
  const f32 = (v: number) => {
    dv.setFloat32(o, v, true)
    o += 4
  }
  const i32 = (v: number) => {
    dv.setInt32(o, v, true)
    o += 4
  }
  f32(p.maxAngle)
  f32(forGen ? p.tcThreshold : 0)
  f32(forGen ? p.stepSize : 0)
  f32(p.relativePeakThresh)
  f32(p.minSeparationAngle)
  f32(p.minSignal)
  i32(seedLo)
  i32(seedHi)
  i32(forGen ? rngOffset : 0)
  i32(nseeds)
  i32(inputs.dims[0])
  i32(inputs.dims[1])
  i32(inputs.dims[2])
  i32(inputs.dims[3])
  i32(inputs.samplmNr)
  i32(inputs.nedges)
  i32(inputs.deltaNr)
  i32(inputs.modelType)
  i32(off.offR)
  i32(off.offDeltaB)
  i32(off.offDeltaQ)
  i32(off.offSampling)
  i32(off.offB0s)
  return buf
}

/** Concatenate typed arrays into one Float32Array (for the packed buffers). */
function concatF32(...arrs: Array<ArrayLike<number>>): Float32Array {
  let n = 0
  for (const a of arrs) n += a.length
  const out = new Float32Array(n)
  let o = 0
  for (const a of arrs) {
    out.set(a as never, o)
    o += a.length
  }
  return out
}

/**
 * Track streamlines from `seeds` (nseeds × 3 voxel coords, flat Float32Array).
 * Returns one Float32Array per surviving streamline (flat x,y,z points, length
 * 3·npts), in VOX space. `onProgress(done, total)` is called per chunk.
 */
export async function trackStreamlines(
  device: GPUDevice,
  inputs: TrackingInputs,
  seeds: Float32Array,
  params: TrackingParams,
  onProgress?: (done: number, total: number) => void,
): Promise<Float32Array[]> {
  const nSeedsTotal = seeds.length / 3
  const off = modelOffsets(inputs)

  // Static buffers (uploaded once), tracked so the finally frees them even if a
  // later allocation or pipeline creation throws. Packed to fit 10 storage
  // buffers/stage:
  //   dataf  = [DWI signal | sphere_vertices]   (verts at offset dimx*dimy*dimz*dimt)
  //   model  = [H | R | delta_b | delta_q | sampling_matrix | b0s_mask(f32)]
  const staticBuffers: GPUBuffer[] = []
  const stat = (b: GPUBuffer): GPUBuffer => {
    staticBuffers.push(b)
    return b
  }
  const out: Float32Array[] = []
  let rngOffset = 0
  const maxBinding = device.limits.maxStorageBufferBindingSize

  try {
    const datafBuf = stat(
      storageFromData(
        device,
        concatF32(inputs.dataf, inputs.sphereVertices),
        'dataf+verts',
      ),
    )
    const metricBuf = stat(
      storageFromData(device, inputs.metricMap, 'metric_map'),
    )
    const edgesBuf = stat(
      storageFromData(device, inputs.sphereEdges, 'sphere_edges'),
    )
    const modelBuf = stat(
      storageFromData(
        device,
        concatF32(
          inputs.H,
          inputs.R,
          inputs.deltaB,
          inputs.deltaQ,
          inputs.samplingMatrix,
          Float32Array.from(inputs.b0sMask),
        ),
        'modelData',
      ),
    )

    // Shader module + the two Boot pipelines.
    const module = device.createShaderModule({ code: bootShaderSource() })
    const getnumPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: BOOT_ENTRY_GETNUM },
    })
    const genPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: BOOT_ENTRY_GEN },
    })

    for (let start = 0; start < nSeedsTotal; start += params.chunkSize) {
      const nseeds = Math.min(params.chunkSize, nSeedsTotal - start)
      const chunk = seeds.subarray(start * 3, (start + nseeds) * 3)
      const gridX = divUp(nseeds, BLOCK_Y)

      // Every per-chunk buffer is tracked so the finally frees it on any path
      // (validation error, readback rejection, throw) — not just success.
      const perChunk: GPUBuffer[] = []
      const track = (b: GPUBuffer): GPUBuffer => {
        perChunk.push(b)
        return b
      }
      try {
        const seedsBuf = track(storageFromData(device, chunk, 'seeds'))
        const offsBuf = track(
          emptyStorage(device, (nseeds + 1) * 4, 'slinesOffs'),
        )
        device.queue.writeBuffer(offsBuf, 0, new Int32Array(nseeds + 1))
        const shDirBytes = inputs.samplmNr * gridX * BLOCK_Y * 3 * REAL_SIZE
        const shDirBuf = track(emptyStorage(device, shDirBytes, 'shDirTemp0'))

        // Pass 1: count streamlines per seed. getNum's auto layout omits
        // metric_map and the output buffers, so they're not bound here.
        const paramsGetnum = track(
          paramsBuffer(
            device,
            packBootParams(inputs, off, params, nseeds, rngOffset, false),
          ),
        )
        dispatchBoot(device, getnumPipe, gridX, [
          bindGroup0(
            device,
            getnumPipe,
            paramsGetnum,
            seedsBuf,
            datafBuf,
            null,
            edgesBuf,
          ),
          bindGroup1(device, getnumPipe, modelBuf),
          bindGroup2(device, getnumPipe, offsBuf, shDirBuf, null),
        ])

        // Exclusive prefix sum of per-seed counts (CPU), written back.
        const counts = new Int32Array(
          await readback(device, offsBuf, (nseeds + 1) * 4),
        )
        const offsets = new Int32Array(nseeds + 1)
        for (let i = 0; i < nseeds; i++) offsets[i + 1] = offsets[i] + counts[i]
        const nSlines = offsets[nseeds]
        device.queue.writeBuffer(offsBuf, 0, offsets)

        if (nSlines > 0) {
          const slineBytes = 2 * 3 * MAX_SLINE_LEN * nSlines * REAL_SIZE
          if (slineBytes > maxBinding) {
            const maxSlines = Math.floor(
              maxBinding / (2 * 3 * MAX_SLINE_LEN * REAL_SIZE),
            )
            throw new Error(
              `this seed batch produced ${nSlines.toLocaleString()} streamlines ` +
                `(${(slineBytes / 1e9).toFixed(1)} GB), over the GPU's ` +
                `${(maxBinding / 1e9).toFixed(1)} GB buffer limit — reduce the seed ` +
                `density or chunk size (max ~${maxSlines.toLocaleString()} streamlines/chunk).`,
            )
          }
          // Interleaved [seed, len] per streamline (seed at 2i, len at 2i+1) — one
          // buffer instead of two, to fit the 10-storage-buffer budget.
          const metaInit = new Int32Array(2 * nSlines)
          for (let i = 0; i < nSlines; i++) metaInit[2 * i] = -1 // seed default -1; len 0
          const slineMetaBuf = track(
            storageFromData(device, metaInit, 'slineMeta'),
          )
          const slineBuf = track(emptyStorage(device, slineBytes, 'sline'))

          // Pass 2: generate streamlines.
          const paramsGen = track(
            paramsBuffer(
              device,
              packBootParams(inputs, off, params, nseeds, rngOffset, true),
            ),
          )
          dispatchBoot(device, genPipe, gridX, [
            bindGroup0(
              device,
              genPipe,
              paramsGen,
              seedsBuf,
              datafBuf,
              metricBuf,
              edgesBuf,
            ),
            bindGroup1(device, genPipe, modelBuf),
            bindGroup2(
              device,
              genPipe,
              offsBuf,
              shDirBuf,
              slineMetaBuf,
              slineBuf,
            ),
          ])

          const slineData = new Float32Array(
            await readback(device, slineBuf, slineBytes),
          )
          const meta = new Int32Array(
            await readback(device, slineMetaBuf, 2 * nSlines * 4),
          )
          const stride = MAX_SLINE_LEN * 2 * 3
          for (let jj = 0; jj < nSlines; jj++) {
            const npts = meta[2 * jj + 1] // odd entries are lengths
            if (npts <= 0 || npts < params.minPts || npts > params.maxPts)
              continue
            out.push(slineData.slice(jj * stride, jj * stride + npts * 3))
          }
        }
      } finally {
        for (const b of perChunk) b.destroy()
      }
      rngOffset += nseeds
      onProgress?.(Math.min(start + nseeds, nSeedsTotal), nSeedsTotal)
    }
  } finally {
    for (const b of staticBuffers) b.destroy()
  }
  return out
}

function paramsBuffer(device: GPUDevice, bytes: ArrayBuffer): GPUBuffer {
  const buf = device.createBuffer({
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
    label: 'params',
  })
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(bytes))
  buf.unmap()
  return buf
}

function dispatchBoot(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  gridX: number,
  groups: GPUBindGroup[],
): void {
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  groups.forEach((g, i) => {
    pass.setBindGroup(i, g)
  })
  pass.dispatchWorkgroups(gridX, 1, 1)
  pass.end()
  device.queue.submit([enc.finish()])
}

// Group 0 (repacked): params(0), seeds(1), dataf(2)=[signal|verts], metric_map(3),
// sphere_edges(4). With layout:'auto', getNum's layout omits metric_map(3) — pass
// metric=null there. Bindings are by explicit index, so gaps are fine.
function bindGroup0(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  params: GPUBuffer,
  seeds: GPUBuffer,
  dataf: GPUBuffer,
  metric: GPUBuffer | null,
  edges: GPUBuffer,
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: params } },
    { binding: 1, resource: { buffer: seeds } },
    { binding: 2, resource: { buffer: dataf } },
    { binding: 4, resource: { buffer: edges } },
  ]
  if (metric) entries.push({ binding: 3, resource: { buffer: metric } })
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  })
}

// Group 1 (repacked): a single modelData buffer.
function bindGroup1(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  modelData: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(1),
    entries: [{ binding: 0, resource: { buffer: modelData } }],
  })
}

// Group 2 (repacked): slineOutOff(0), shDir0(1), slineMeta(2), sline(3).
// getNum binds only 0+1; gen binds slineMeta + sline too.
function bindGroup2(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  offs: GPUBuffer,
  shDir: GPUBuffer,
  slineMeta: GPUBuffer | null,
  sline?: GPUBuffer | null,
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: offs } },
    { binding: 1, resource: { buffer: shDir } },
  ]
  if (slineMeta) entries.push({ binding: 2, resource: { buffer: slineMeta } })
  if (sline) entries.push({ binding: 3, resource: { buffer: sline } })
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(2),
    entries,
  })
}
