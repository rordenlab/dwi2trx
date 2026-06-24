/**
 * Generic drag-and-drop file collection — walks a DataTransfer, recursing into
 * dropped folders. Lives here (not in the dcm2niix wrapper) so the DICOM WASM
 * stays lazily imported: every drop walks files, but only DICOM drops pull in
 * dcm2niix.
 *
 * Each file is stamped with `_webkitRelativePath` so dcm2niix can group by
 * series. `webkitGetAsEntry` is non-standard (Chromium/WebKit); callers fall
 * back to `DataTransfer.files` when it's absent.
 */

type FileWithRelativePath = File & { _webkitRelativePath?: string }

export async function traverseDataTransferItems(
  items: DataTransferItemList,
  limit = Number.POSITIVE_INFINITY,
): Promise<File[]> {
  const files: File[] = []
  const entries: FileSystemEntry[] = []
  for (let i = 0; i < items.length; i++) {
    const entry = items[i].webkitGetAsEntry()
    if (entry) entries.push(entry)
  }
  await Promise.all(entries.map((entry) => walkEntry(entry, '', files, limit)))
  return files
}

function walkEntry(
  entry: FileSystemEntry,
  path: string,
  out: File[],
  limit: number,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (out.length > limit) {
      reject(new Error(`Too many files (limit ${limit}).`))
      return
    }
    if (entry.isFile) {
      ;(entry as FileSystemFileEntry).file((file) => {
        if (out.length >= limit) {
          reject(new Error(`Too many files (limit ${limit}).`))
          return
        }
        const tagged = file as FileWithRelativePath
        tagged._webkitRelativePath = path + file.name
        out.push(tagged)
        resolve()
      }, reject)
      return
    }
    if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      const childPath = `${path}${entry.name}/`
      const readBatch = () => {
        reader.readEntries((batch) => {
          if (batch.length === 0) {
            resolve()
            return
          }
          Promise.all(
            batch.map((child) => walkEntry(child, childPath, out, limit)),
          )
            .then(readBatch)
            .catch(reject)
        }, reject)
      }
      readBatch()
      return
    }
    resolve()
  })
}
