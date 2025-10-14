<div align="center">

# @chatluna/luna-vdb

_轻量级，本地的 Wasm 向量数据库。_

## [![npm](https://img.shields.io/npm/v/@chatluna/luna-vdb)](https://www.npmjs.com/package/@chatluna/luna-vdb) [![npm](https://img.shields.io/npm/dm/@chatluna/luna-vdb)](https://www.npmjs.com/package/@chatluna/luna-vdb) ![node version](https://img.shields.io/badge/node-%3E=18-green) ![github top language](https://img.shields.io/github/languages/top/ChatLunaLab/luna-vdb?logo=github)

</div>

## 特性

- Rust + WebAssembly，浏览器和 Node.js 均可使用
- 简单，灵活的 API，涵盖向量索引、增量更新、搜索与清空。使用 ID 标注文档。
- 支持序列化/反序列化
- TypeScript 类型支持

## 快速开始

### 安装

```bash
yarn install @chatluna/luna-vdb
```

### API

```ts
import { LunaVDB } from "@chatluna/luna-vdb";

const engine = new LunaVDB();

// 初始化一组向量
engine.index({
  embeddings: [
    { id: "cat", embeddings: [0.8, 0.7, 0.6] },
    { id: "dog", embeddings: [0.4, 0.3, 0.2] },
  ],
});

// 搜索最相近的两个向量
const result = engine.search(new Float32Array([0.75, 0.65, 0.55]), 2);
result.neighbors.forEach(({ id, distance }) => {
  console.log(`${id}: ${distance}`);
});

// 增量更新
engine.add({
  embeddings: [{ id: "bird", embeddings: [0.2, 0.1, 0.9] }],
});

// 删除向量或清空索引
engine.remove(["dog"]);
// engine.clear();

// 序列化以便持久化
const snapshot = engine.serialize(); // 返回 ArrayBuffer
const restored = LunaVDB.deserialize(snapshot);

console.log(restored.size()); // => 当前向量数量
```

> 在浏览器环境中以相同方式使用，只需通过 `import`/`await import` 加载打包后的 WASM 模块即可。

## 感谢

- [tinyvector](https://github.com/m1guelpf/tinyvector)
