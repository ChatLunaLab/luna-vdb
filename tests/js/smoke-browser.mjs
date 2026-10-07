// Browser-bundle smoke test.
//
// Exercises the `pkg/web` and `pkg/web-scalar` builds through their real ESM
// entry points, the same way a bundler would, including `init()`.
//
// It deliberately does NOT assert anything about SIMD opcodes.
//
// It used to: a hand-rolled scan walked the code section looking for the `0xFD`
// prefix followed by a vector sub-opcode. That scan cannot tell an instruction
// from an immediate operand, so it reported 26 "SIMD opcode candidates" in a
// scalar artifact that `wasm-tools print` shows contains zero vector
// instructions — the `0xfd` bytes were inside LEB128 constants and float
// literals. A checker that cries wolf on a correct build is worse than no
// checker, because the next real regression gets waved through as noise.
//
// The opcode assertion now lives in the CI package job, where `wasm-tools`
// decodes the module properly and the check runs against the packaged
// `luna_vdb_bg.wasm` — the artifact that actually ships, after wasm-bindgen and
// wasm-opt, rather than the intermediate cargo output.

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
// Tests
// ---------------------------------------------------------------------------

console.log('luna-vdb browser-bundle smoke test')

// SIMD128 is not runtime-detectable the way AVX2 is: a module compiled with
// `+simd128` fails to *instantiate* on an engine without it. That makes
// instantiation itself the strongest check available from JS — a scalar module
// that picked up vector instructions could not get this far on a runtime that
// lacks them. Where the host *does* support SIMD, the CI opcode check is the
// one that can actually tell the two artifacts apart.
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

  // The generated bindings this module backs must be reachable from the
  // packaged entry point.
  const exported = names.join(' ')
  assert.ok(
    exported.includes('search') || exported.includes('Luna'),
    `expected the LunaVDB exports, found: ${exported}`,
  )
})

await test('scalar module is loadable through its ESM entry point', async () => {
  const entry = path.join(pkgRoot, 'web-scalar', 'luna_vdb.js')
  const module = await import(`file://${entry}`)

  // `--target web` emits an `init` function that must be awaited before the
  // handle is usable; if the packaging dropped it, every consumer breaks.
  assert.equal(typeof module.default, 'function', 'expected a default init export')
  await module.default({
    module_or_path: await readFile(path.join(pkgRoot, 'web-scalar', 'luna_vdb_bg.wasm')),
  })

  assert.equal(typeof module.LunaVDB, 'function')
  assert.equal(module.simdBackend(), 'scalar')
  const db = new module.LunaVDB()
  assert.equal(db.size(), 0)
})

await test('simd module validates and loads through its ESM entry point', async () => {
  const wasmPath = path.join(pkgRoot, 'web', 'luna_vdb_bg.wasm')
  const bytes = await readFile(wasmPath)

  // CI runs Node 22, which supports SIMD128, so a module that fails to
  // validate here is broken — not "an old runtime". (This used to log a
  // warning and count as a pass.)
  assert.ok(WebAssembly.validate(bytes), 'the SIMD build does not validate')

  const module = await import(`file://${path.join(pkgRoot, 'web', 'luna_vdb.js')}`)
  await module.default({ module_or_path: bytes })
  assert.equal(module.simdBackend(), 'wasm-simd128')
  const db = new module.LunaVDB()
  db.add({ embeddings: [{ id: 'a', embeddings: [1, 0] }] })
  assert.equal(db.search([1, 0], 1).neighbors[0].id, 'a')
})

await test('node scalar build loads and reports the scalar kernel', async () => {
  const { createRequire } = await import('node:module')
  const require = createRequire(import.meta.url)
  const scalar = require(path.join(pkgRoot, 'nodejs-scalar', 'luna_vdb.js'))
  assert.equal(scalar.simdBackend(), 'scalar')
  const db = new scalar.LunaVDB()
  db.add({ embeddings: [{ id: 'a', embeddings: [1, 0] }] })
  assert.equal(db.search([1, 0], 1).neighbors[0].id, 'a')
})

console.log(`\n${passed} passed, ${failed} failed`)

if (failed > 0) {
  process.exit(1)
}
