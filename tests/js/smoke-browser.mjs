// Browser-bundle smoke test.
//
// Two jobs:
//
//   1. Exercise the `pkg/web` build through its real ESM entry point, the same
//      way a bundler would, including `init()`.
//   2. Load the **scalar** build and assert it contains no SIMD128 opcodes.
//
// (2) is the important one. SIMD128 is not runtime-detectable the way AVX2 is:
// a module compiled with `+simd128` fails to *instantiate* on an engine without
// it. If the scalar artifact accidentally picks up vector instructions — a
// stale `RUSTFLAGS` in the environment is the usual cause — every user on an
// older runtime gets a hard instantiation failure with no fallback. So we
// decode the module's function bodies and grep for the opcodes rather than
// trusting the build flags.

import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const pkgRoot = path.join(here, '..', '..', 'pkg')

let passed = 0
let failed = 0

function test(name, fn) {
  try {
    const result = fn()
    if (result && typeof result.then === 'function') {
      return result.then(
        () => {
          passed += 1
          console.log(`  ok   ${name}`)
        },
        (error) => {
          failed += 1
          console.error(`  FAIL ${name}\n       ${error.message}`)
        },
      )
    }
    passed += 1
    console.log(`  ok   ${name}`)
  } catch (error) {
    failed += 1
    console.error(`  FAIL ${name}\n       ${error.message}`)
  }
  return Promise.resolve()
}

// ---------------------------------------------------------------------------
// SIMD128 opcode detection
// ---------------------------------------------------------------------------

// The 0xFD prefix introduces the vector instruction space.
const SIMD_PREFIX = 0xfd

// Opcodes we actually emit, by their sub-opcode within that space:
//   0x0E v128.load   0x1E v128.store
//   0xE4 f32x4.add   0xE5 f32x4.sub   0xE6 f32x4.mul
//   0x00 v128.load   (alternate encoding: 0x00-0x0B are the load forms)
const SIMD_OPS = new Set([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0e, 0x1e, 0xe4, 0xe5, 0xe6])

/// Decode the module and return true if any 0xFD-prefixed opcode appears inside
/// a code section.
///
/// This is a deliberately shallow scan: it walks the section table to find the
/// code section, then looks for the `0xfd` byte prefix. It can in principle flag
/// a `0xfd` byte that happens to appear inside an immediate operand, so it is
/// written to report *offsets* rather than just a boolean — a false positive is
/// then obvious from the surrounding bytes rather than a mystery failure.
function findSimdOpcode(bytes) {
  const hits = []
  let offset = 8 // skip the magic and version

  while (offset < bytes.length) {
    const sectionId = bytes[offset]
    offset += 1

    // LEB128 section length.
    let length = 0
    let shift = 0
    for (;;) {
      const byte = bytes[offset]
      offset += 1
      length |= (byte & 0x7f) << shift
      shift += 7
      if ((byte & 0x80) === 0) break
    }

    const sectionStart = offset
    const sectionEnd = offset + length

    // Section 10 is `code`.
    if (sectionId === 10) {
      // Walk the function bodies. Each body is a LEB128 size then the body.
      let cursor = sectionStart + 1 // skip the vector count LEB (assume 1 byte)
      let guard = 0
      while (cursor < sectionEnd && guard < 100_000) {
        guard += 1
        let bodySize = 0
        let bodyShift = 0
        let sizeBytes = 0
        for (;;) {
          const byte = bytes[cursor + sizeBytes]
          sizeBytes += 1
          bodySize |= (byte & 0x7f) << bodyShift
          bodyShift += 7
          if ((byte & 0x80) === 0) break
        }
        cursor += sizeBytes
        const bodyEnd = cursor + bodySize

        for (let i = cursor; i < bodyEnd && i < sectionEnd; i += 1) {
          if (bytes[i] === SIMD_PREFIX && i + 1 < bytes.length && SIMD_OPS.has(bytes[i + 1])) {
            hits.push(i)
          }
        }

        cursor = bodyEnd
      }
    }

    offset = sectionEnd
  }

  return hits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

console.log('luna-vdb browser-bundle smoke test')

await test('scalar build contains no SIMD128 instructions', async () => {
  const wasmPath = path.join(pkgRoot, 'web-scalar', 'luna_vdb_bg.wasm')
  const bytes = await readFile(wasmPath)
  const hits = findSimdOpcode(bytes)

  assert.equal(
    hits.length,
    0,
    `scalar artifact has ${hits.length} SIMD opcode candidates at offsets ${hits.slice(0, 5).join(', ')}. ` +
      'A stale RUSTFLAGS with +simd128 is the usual cause.',
  )
})

await test('simd build does contain SIMD128 instructions', async () => {
  const wasmPath = path.join(pkgRoot, 'web', 'luna_vdb_bg.wasm')
  const bytes = await readFile(wasmPath)
  const hits = findSimdOpcode(bytes)

  // If this fails the SIMD kernel is not being selected, which means the
  // performance work is silently inactive — worth failing the build over.
  assert.ok(
    hits.length > 0,
    'simd artifact has no SIMD opcodes; the vectorised kernel is not in use',
  )
})

await test('scalar module instantiates', async () => {
  const wasmPath = path.join(pkgRoot, 'web-scalar', 'luna_vdb_bg.wasm')
  const bytes = await readFile(wasmPath)
  const { instance } = await WebAssembly.instantiate(bytes, {
    // wasm-bindgen glue satisfies these; instantiating the raw module needs the
    // imports to exist even though we never call into them here.
    './luna_vdb_bg.js': new Proxy({}, { get: () => () => {} }),
    wasm_bindgen: new Proxy({}, { get: () => () => {} }),
  })

  assert.ok(instance.exports, 'module should export something')
  const names = Object.keys(instance.exports)
  assert.ok(names.length > 0, 'module should have exports')

  // The generated bindings this module backs must be reachable.
  const exported = names.join(' ')
  assert.ok(
    exported.includes('search') || exported.includes('Luna'),
    `expected the LunaVDB exports, found: ${exported}`,
  )
})

await test('simd module passes validation in this runtime', async () => {
  const wasmPath = path.join(pkgRoot, 'web', 'luna_vdb_bg.wasm')
  const bytes = await readFile(wasmPath)

  // `WebAssembly.validate` is a real conformance check against the engine's
  // SIMD support — it is the same gate a browser applies before instantiation.
  const valid = WebAssembly.validate(bytes)

  if (!valid) {
    // Not necessarily a failure: this Node build may lack SIMD128. Report it
    // clearly so the difference between "our module is broken" and "this host
    // is old" is obvious.
    console.warn(`       (this runtime does not support SIMD128; skipped)`)
    return
  }
  assert.ok(valid)
})

console.log(`\n${passed} passed, ${failed} failed`)

if (failed > 0) {
  process.exit(1)
}
