#!/usr/bin/env tsx

import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'

const args = process.argv.slice(2)

if (!args.includes('--otp')) {
  console.error('请提供 --otp 参数')
  process.exit(1)
}

const otpIndex = args.indexOf('--otp')
const otp = args[otpIndex + 1]

const currentDirectory = process.cwd()

const packageJsonPath = path.join(currentDirectory, 'package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const packageName = packageJson.name

const publishCommand = `npm publish --access public --otp=${otp}`

try {
  execSync(publishCommand, { cwd: currentDirectory, stdio: 'inherit' })

  const syncCommand = `https://registry-direct.npmmirror.com/${packageName}/sync?sync_upstream=true`
  fetch(syncCommand, { method: 'PUT' })
    .then((resp) => {
      if (resp.status !== 200) {
        throw new Error('同步失败')
      }
      console.log('同步成功')
    })
    .catch(() => {})
} catch (error) {
  console.error('发布失败:', (error as Error).message)
  process.exit(1)
}
