#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const semver = require('semver');

const args = process.argv.slice(2);

// 解析命令行参数
const options = {
  major: args.includes('-1') || args.includes('--major'),
  minor: args.includes('-2') || args.includes('--minor'),
  patch: args.includes('-3') || args.includes('--patch'),
  prerelease: args.includes('-p') || args.includes('--prerelease'),
  version: args.find(arg => arg.startsWith('-v') || arg.startsWith('--version')),
  recursive: args.includes('-r') || args.includes('--recursive'),
};

// 读取 package.json 文件
const packageJsonPath = path.join(process.cwd(), 'package.json');
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

// 更新版本号
let newVersion;
if (options.version) {
  newVersion = options.version;
} else if (options.prerelease) {
  const currentVersion = packageJson.version;
  const parsedVersion = semver.parse(currentVersion);
  if (parsedVersion.prerelease[0] === 'alpha') {
    newVersion = semver.inc(currentVersion, 'prerelease', 'beta');
  } else if (parsedVersion.prerelease[0] === 'beta') {
    newVersion = semver.inc(currentVersion, 'prerelease', 'rc');
  } else if (parsedVersion.prerelease[0] === 'rc') {
    newVersion = semver.inc(currentVersion, 'patch');
  } else {
    newVersion = semver.inc(currentVersion, 'prerelease', 'alpha');
  }
} else if (options.major) {
  newVersion = semver.inc(packageJson.version, 'major');
} else if (options.minor) {
  newVersion = semver.inc(packageJson.version, 'minor');
} else if (options.patch) {
  newVersion = semver.inc(packageJson.version, 'patch');
} else {
  newVersion = semver.inc(packageJson.version, 'patch');
}

packageJson.version = newVersion;

// 写入更新后的 package.json 文件
fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));

console.log(`版本号已更新为: ${newVersion}`);
