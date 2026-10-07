// JS-level smoke test for the published Node package.
//
// Runs against `pkg/nodejs` — the real generated bindings, not the Rust API —
// so it catches the class of breakage that only appears once wasm-bindgen has
// been involved: bad glue, a missing `.wasm` path, a type that does not survive
// `into_wasm_abi`.
//
// Every check in the "errors" section is a regression guard. The old bindings
// called `.unwrap()` in `add`, `remove`, `serialize` and `deserialize`, so each
// of these inputs either killed the process or poisoned the instance and made
// the *next* call return garbage. They must now be ordinary catchable throws.

import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { LunaVDB, simdBackend, version, isSnapshot, snapshotVersion } = require('../../pkg/nodejs/luna_vdb.js')

let passed = 0
let failed = 0

function test(name, fn) {
  try {
    fn()
    passed += 1
    console.log(`  ok   ${name}`)
  } catch (error) {
    failed += 1
    console.error(`  FAIL ${name}`)
    console.error(`       ${error && error.message ? error.message : error}`)
  }
}

const dim = 8

function vector(seed) {
  // Mix the seed first (murmur3's finaliser, a bijection on 32 bits). The
  // obvious `seed | 1` maps 2n and 2n + 1 to the same state, so every corpus
  // used to contain identical twin vectors, and a "nearest neighbour is
  // itself" check only tested which twin the tie-break happened to prefer.
  let state = Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b) >>> 0
  state ^= state >>> 13
  state = Math.imul(state, 0xc2b2ae35) >>> 0
  state ^= state >>> 16
  if (state === 0) state = 1
  const out = new Float32Array(dim)
  for (let i = 0; i < dim; i += 1) {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    out[i] = (state / 0xffffffff) - 0.5
  }
  return out
}

function makeResource(count, prefix = 'v') {
  const embeddings = []
  for (let i = 0; i < count; i += 1) {
    embeddings.push({ id: `${prefix}${i}`, embeddings: Array.from(vector(i + 1)) })
  }
  return { embeddings }
}

console.log(`luna-vdb smoke test (version ${version()}, simd backend ${simdBackend()})`)

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

test('constructs empty', () => {
  const db = new LunaVDB()
  assert.equal(db.size(), 0)
  assert.equal(db.dimension(), 0)
  assert.equal(db.distance(), 'euclidean')
})

test('constructs with options', () => {
  const db = new LunaVDB({ distance: 'cosine', nprobe: 4 })
  assert.equal(db.distance(), 'cosine')
})

test('rejects an unknown distance', () => {
  assert.throws(() => new LunaVDB({ distance: 'hamming' }), /unknown distance/)
})

test('survives a thrown constructor error', () => {
  // The instance must be usable afterwards — this is the poisoning check in
  // its simplest form.
  assert.throws(() => new LunaVDB({ distance: 'nope' }))
  const db = new LunaVDB()
  assert.equal(db.size(), 0)
})

// ---------------------------------------------------------------------------
// Index / add / search
// ---------------------------------------------------------------------------

test('indexes and searches', () => {
  const db = new LunaVDB()
  db.index(makeResource(100))

  assert.equal(db.size(), 100)
  assert.equal(db.dimension(), dim)

  const result = db.search(Array.from(vector(1)), 5)
  assert.equal(result.neighbors.length, 5)
  assert.equal(result.neighbors[0].id, 'v0')
  assert.ok(result.neighbors[0].distance < 1e-5, `distance was ${result.neighbors[0].distance}`)

  for (let i = 1; i < result.neighbors.length; i += 1) {
    assert.ok(
      result.neighbors[i - 1].distance <= result.neighbors[i].distance,
      'distances must be ascending',
    )
  }
})

test('accepts a Float32Array query', () => {
  const db = new LunaVDB()
  db.index(makeResource(50))
  const result = db.search(vector(3), 3)
  assert.equal(result.neighbors.length, 3)
})

test('search reports its own work', () => {
  const db = new LunaVDB({ ivfThreshold: 256 })
  db.index(makeResource(2000))

  const result = db.search(Array.from(vector(10)), 10)
  assert.equal(typeof result.scanned, 'number')
  assert.equal(typeof result.rescored, 'number')
  assert.equal(typeof result.cellsProbed, 'number')
  assert.equal(typeof result.exact, 'boolean')
  assert.ok(result.scanned <= 2000)
})

test('adds incrementally', () => {
  const db = new LunaVDB()
  db.add(makeResource(10))
  assert.equal(db.size(), 10)

  db.add({ embeddings: [{ id: 'extra', embeddings: Array.from(vector(99)) }] })
  assert.equal(db.size(), 11)
  assert.ok(db.has('extra'))

  const result = db.search(Array.from(vector(99)), 1)
  assert.equal(result.neighbors[0].id, 'extra')
})

test('removes', () => {
  const db = new LunaVDB()
  db.add(makeResource(10))
  db.remove(['v0', 'v1'])
  assert.equal(db.size(), 8)
  assert.ok(!db.has('v0'))
})

test('clears', () => {
  const db = new LunaVDB()
  db.add(makeResource(10))
  db.clear()
  assert.equal(db.size(), 0)
  assert.equal(db.dimension(), 0)

  // Still usable after clearing.
  db.add({ embeddings: [{ id: 'a', embeddings: Array.from(vector(5)) }] })
  assert.equal(db.size(), 1)
})

// ---------------------------------------------------------------------------
// Errors — the "神秘的空指针" regression set
// ---------------------------------------------------------------------------

test('duplicate id throws a real Error', () => {
  const db = new LunaVDB()
  db.add({ embeddings: [{ id: 'dup', embeddings: Array.from(vector(1)) }] })

  let caught = null
  try {
    db.add({ embeddings: [{ id: 'dup', embeddings: Array.from(vector(2)) }] })
  } catch (error) {
    caught = error
  }

  assert.ok(caught, 'expected a throw')
  assert.ok(caught instanceof Error, `expected an Error, got ${typeof caught}`)
  assert.match(caught.message, /already exists/)
  assert.ok(caught.stack, 'the stack must survive so the failure is traceable')
})

test('dimension mismatch throws and keeps the instance usable', () => {
  const db = new LunaVDB()
  db.add({ embeddings: [{ id: 'a', embeddings: Array.from(vector(1)) }] })

  assert.throws(
    () => db.add({ embeddings: [{ id: 'b', embeddings: [1, 2, 3] }] }),
    /dimension mismatch/,
  )

  // The critical assertion: the *next* call still works.
  assert.equal(db.size(), 1)
  assert.equal(db.search(Array.from(vector(1)), 1).neighbors.length, 1)
})

test('removing an unknown id throws and keeps the instance usable', () => {
  const db = new LunaVDB()
  db.add(makeResource(5))

  assert.throws(() => db.remove(['does-not-exist']), /not found/)

  assert.equal(db.size(), 5)
  assert.equal(db.search(Array.from(vector(1)), 1).neighbors.length, 1)
})

test('deserialize rejects corrupt payloads without poisoning the instance', () => {
  const db = new LunaVDB()
  db.index(makeResource(200, 'c'))
  const good = db.serialize()

  // Truncations at every structurally interesting offset.
  for (const cut of [0, 1, 4, 8, 15, 16, 17, Math.floor(good.length / 2), good.length - 1]) {
    assert.throws(
      () => LunaVDB.deserialize(good.slice(0, cut)),
      (error) => error instanceof Error,
      `truncation to ${cut} should throw an Error`,
    )
  }

  // Bit flips in the payload must be caught by the checksum.
  for (const offset of [16, 32, Math.floor(good.length / 2), good.length - 1]) {
    if (offset >= good.length) continue
    const corrupted = good.slice()
    corrupted[offset] ^= 0xff
    assert.throws(
      () => LunaVDB.deserialize(corrupted),
      (error) => error instanceof Error,
      `bit flip at ${offset} should throw an Error`,
    )
  }

  // Unrelated bytes.
  assert.throws(() => LunaVDB.deserialize(new Uint8Array([1, 2, 3, 4, 5])))
  assert.throws(() => LunaVDB.deserialize(new Uint8Array(0)))

  // The original handle is untouched throughout.
  assert.equal(db.size(), 200)
  assert.equal(db.search(Array.from(vector(1)), 5).neighbors.length, 5)
})

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

test('serialize / deserialize round-trips', () => {
  const db = new LunaVDB()
  db.index(makeResource(300, 'p'))

  const bytes = db.serialize()
  assert.ok(bytes instanceof Uint8Array, `expected Uint8Array, got ${bytes?.constructor?.name}`)
  assert.ok(bytes.length > 0)
  assert.ok(isSnapshot(bytes))
  assert.equal(snapshotVersion(bytes), 2)

  const restored = LunaVDB.deserialize(bytes)
  assert.equal(restored.size(), 300)
  assert.equal(restored.dimension(), dim)
  assert.equal(restored.distance(), db.distance())

  // Results must be identical, not merely close.
  for (const seed of [1, 50, 200]) {
    const before = db.search(Array.from(vector(seed)), 10)
    const after = restored.search(Array.from(vector(seed)), 10)
    assert.deepEqual(after.neighbors, before.neighbors)
  }
})

test('uncompressed round-trips too', () => {
  const db = new LunaVDB()
  db.index(makeResource(150, 'u'))

  const raw = db.serialize(false)
  const packed = db.serialize(true)

  // Compression should not make small payloads bigger, but for a payload this
  // small either may win; both must load.
  const fromRaw = LunaVDB.deserialize(raw)
  const fromPacked = LunaVDB.deserialize(packed)
  assert.equal(fromRaw.size(), 150)
  assert.equal(fromPacked.size(), 150)

  const query = Array.from(vector(7))
  assert.deepEqual(fromRaw.search(query, 5).neighbors, fromPacked.search(query, 5).neighbors)
})

test('round-trips after mutation', () => {
  const db = new LunaVDB({ ivfThreshold: 256 })
  db.index(makeResource(600, 'm'))
  db.add(makeResource(50, 'late'))
  db.remove(Array.from({ length: 40 }, (_, i) => `m${i}`))

  const restored = LunaVDB.deserialize(db.serialize())
  assert.equal(restored.size(), db.size())

  const query = Array.from(vector(1))
  assert.deepEqual(restored.search(query, 10).neighbors, db.search(query, 10).neighbors)
})

test('restoreInto reuses the handle', () => {
  const source = new LunaVDB()
  source.index(makeResource(80, 'r'))
  const bytes = source.serialize()

  const target = new LunaVDB()
  target.restoreInto(bytes)
  assert.equal(target.size(), 80)

  // A bad payload must not wipe the existing contents.
  assert.throws(() => target.restoreInto(new Uint8Array([9, 9, 9])))
  assert.equal(target.size(), 80)
})

test('empty engine round-trips', () => {
  const db = new LunaVDB()
  const restored = LunaVDB.deserialize(db.serialize())
  assert.equal(restored.size(), 0)
})

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

test('stats are populated', () => {
  const db = new LunaVDB({ ivfThreshold: 256 })
  db.index(makeResource(1000))

  const stats = db.stats()
  assert.equal(stats.size, 1000)
  assert.equal(stats.dimension, dim)
  assert.equal(stats.distance, 'euclidean')
  assert.ok(stats.indexed)
  assert.ok(stats.nlist > 0)
  assert.ok(stats.memoryBytes > 0)
  assert.equal(stats.pendingDeletes, 0)
  assert.ok(typeof stats.simd === 'string' && stats.simd.length > 0)
})

test('searchExact is exact', () => {
  const db = new LunaVDB({ ivfThreshold: 256 })
  db.index(makeResource(1000, 'e'))

  const exact = db.searchExact(Array.from(vector(5)), 5)
  assert.equal(exact.exact, true)
  assert.equal(exact.scanned, 1000)
  assert.equal(exact.neighbors[0].id, 'e4')
})

test('default search is exact', () => {
  const db = new LunaVDB({ ivfThreshold: 256 })
  db.index(makeResource(3000, 'x'))
  assert.ok(db.stats().indexed)

  for (const seed of [10_001, 10_002, 10_003, 10_004, 10_005]) {
    const query = Array.from(vector(seed))
    const fast = db.search(query, 10)
    const slow = db.searchExact(query, 10)
    assert.equal(fast.exact, true)
    assert.deepEqual(fast.neighbors.map((n) => n.id), slow.neighbors.map((n) => n.id))
  }
})

test('approximate mode is opt-in and says so', () => {
  const db = new LunaVDB({ ivfThreshold: 256, approximate: true, nprobe: 2 })
  db.index(makeResource(3000, 'a'))
  const result = db.search(Array.from(vector(7)), 10)
  assert.equal(result.exact, false)
  assert.ok(result.scanned < 3000)
  assert.equal(db.stats().approximate, true)
})

test('a malformed argument throws and does not lock the handle', () => {
  // REGRESSION: wasm-bindgen held the handle's borrow while converting the
  // argument, and a failed conversion never released it — every later call
  // threw "recursive use of an object detected", and free() failed too.
  const db = new LunaVDB()
  db.add(makeResource(5))

  assert.throws(() => db.add({ embeddings: [{ id: 42, embeddings: [1, 2, 3, 4, 5, 6, 7, 8] }] }), TypeError)
  assert.throws(() => db.add('not a resource'), TypeError)
  assert.throws(() => db.add(undefined), TypeError)
  assert.throws(() => db.index(null), TypeError)
  assert.throws(() => db.remove('v0'), TypeError)
  assert.throws(() => db.remove([1, 2]), TypeError)
  assert.throws(() => db.restoreInto('bytes'), TypeError)
  assert.throws(() => db.search([NaN, 0, 0, 0, 0, 0, 0, 0], 1), TypeError)

  // Every kind of call still works on the same handle — and it can be freed.
  assert.equal(db.size(), 5)
  assert.equal(db.search(Array.from(vector(1)), 2).neighbors.length, 2)
  db.add({ embeddings: [{ id: 'late', embeddings: Array.from(vector(77)) }] })
  db.remove(['late'])
  db.clear()
  assert.equal(db.size(), 0)
  db.free()
})

test('batches are all-or-nothing', () => {
  const db = new LunaVDB()
  db.add(makeResource(3))

  const batch = makeResource(4, 'b')
  batch.embeddings[2].embeddings = [1, 2, 3]
  assert.throws(() => db.add(batch), /dimension mismatch/)
  assert.equal(db.size(), 3)
  assert.ok(!db.has('b0'), 'items before the bad one must not be kept')

  assert.throws(() => db.remove(['v0', 'missing']), /not found/)
  assert.ok(db.has('v0'), 'nothing is removed when one id is unknown')
})

test('k larger than the corpus, or negative, is clamped', () => {
  const db = new LunaVDB()
  db.add(makeResource(4))
  assert.equal(db.search(Array.from(vector(1)), 1000).neighbors.length, 4)
  // -1 crosses the boundary as 2^32 - 1.
  assert.equal(db.search(Array.from(vector(1)), -1).neighbors.length, 4)
})

test('deserialize accepts an ArrayBuffer and a Buffer', () => {
  // REGRESSION: an ArrayBuffer has no `length`, so the old binding read it as
  // zero bytes and called a valid snapshot truncated.
  const db = new LunaVDB({ distance: 'cosine' })
  db.index(makeResource(40, 'ab'))
  const bytes = db.serialize()
  const exact = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)

  assert.equal(LunaVDB.deserialize(exact).size(), 40)
  assert.equal(LunaVDB.deserialize(Buffer.from(bytes)).size(), 40)
  assert.ok(isSnapshot(exact))

  const target = new LunaVDB()
  target.restoreInto(exact)
  assert.equal(target.size(), 40)
  // The metric comes from the snapshot and survives a rebuild.
  target.index(makeResource(5, 'z'))
  assert.equal(target.distance(), 'cosine')

  assert.throws(() => LunaVDB.deserialize(42), TypeError)
})

test('every header byte is covered by the checksum', () => {
  const db = new LunaVDB()
  db.index(makeResource(30, 'h'))
  const good = db.serialize()
  for (let offset = 4; offset < 16; offset += 1) {
    const bad = good.slice()
    bad[offset] ^= 0x01
    assert.throws(() => LunaVDB.deserialize(bad), Error, `flip at ${offset}`)
  }
})

test('helpers report format information', () => {
  assert.equal(isSnapshot(new Uint8Array([1, 2, 3])), false)
  assert.equal(snapshotVersion(new Uint8Array([1, 2, 3])), 0)
})

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failed} failed`)

if (failed > 0) {
  process.exit(1)
}
