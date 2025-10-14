#!/usr/bin/env tsx

import fs from 'fs'
import path from 'path'
import semver from 'semver'

const args = process.argv.slice(2)

const options = {
  major: args.includes('-1') || args.includes('--major'),
  minor: args.includes('-2') || args.includes('--minor'),
  patch: args.includes('-3') || args.includes('--patch'),
  prerelease: args.includes('-p') || args.includes('--prerelease'),
  version: args.find(arg => arg.startsWith('-v') || arg.startsWith('--version')),
  recursive: args.includes('-r') || args.includes('--recursive'),
}

const packageJsonPath = path.join(process.cwd(), 'package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))

let newVersion: string | null
if (options.version) {
  newVersion = options.version
} else if (options.prerelease) {
  const currentVersion = packageJson.version
  const parsedVersion = semver.parse(currentVersion)
  if (parsedVersion?.prerelease[0] === 'alpha') {
    newVersion = semver.inc(currentVersion, 'prerelease', 'beta')
  } else if (parsedVersion?.prerelease[0] === 'beta') {
    newVersion = semver.inc(currentVersion, 'prerelease', 'rc')
  } else if (parsedVersion?.prerelease[0] === 'rc') {
    newVersion = semver.inc(currentVersion, 'patch')
  } else {
    newVersion = semver.inc(currentVersion, 'prerelease', 'alpha')
  }
} else if (options.major) {
  newVersion = semver.inc(packageJson.version, 'major')
} else if (options.minor) {
  newVersion = semver.inc(packageJson.version, 'minor')
} else if (options.patch) {
  newVersion = semver.inc(packageJson.version, 'patch')
} else {
  newVersion = semver.inc(packageJson.version, 'patch')
}

packageJson.version = newVersion

fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2))

console.log(`版本号已更新为: ${newVersion}`)
