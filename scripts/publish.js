#!/usr/bin/env node

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

// 获取命令行参数
const args = process.argv.slice(2);
// 检查是否提供了 --otp 参数
if (!args.includes("--otp")) {
    console.error("请提供 --otp 参数");
    process.exit(1);
}

// 获取 --otp 参数的值
const otpIndex = args.indexOf("--otp");
const otp = args[otpIndex + 1];

// 获取当前目录的路径
const currentDirectory = process.cwd();

// 读取 package.json 文件
const packageJsonPath = path.join(currentDirectory, "package.json");
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
const packageName = packageJson.name;

// 构建发布命令
const publishCommand = `npm publish --access public --otp=${otp}`;

try {
    // 执行发布命令
    execSync(publishCommand, { cwd: currentDirectory, stdio: "inherit" });

    // 同步到 npmmirror
    const syncCommand = `https://registry-direct.npmmirror.com/${packageName}/sync?sync_upstream=true`;
    fetch(syncCommand, { method: "PUT" })
        .then((resp) => {
            if (resp.status !== 200) {
                throw new Error("同步失败");
            }
            console.log("同步成功");
        })
        .catch(() => {});
} catch (error) {
    console.error("发布失败:", error.message);
    process.exit(1);
}
