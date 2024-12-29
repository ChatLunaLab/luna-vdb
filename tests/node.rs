extern crate wasm_bindgen_test;
use luna_vdb::*;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn test_luna_vdb_search() {
    // 初始化LunaVDB实例
    let mut luna_vdb = LunaVDB::new(None);

    // 创建一些测试数据
    let embeddings = vec![
        EmbeddedResource {
            id: "1".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
        },
        EmbeddedResource {
            id: "2".to_string(),
            embeddings: vec![0.4, 0.5, 0.6],
        },
    ];
    let resource = Resource { embeddings };

    // 索引数据
    luna_vdb.index(resource);

    // 执行搜索
    let query = vec![0.15, 0.25, 0.35];
    let neighbors = luna_vdb.search(query, 1);

    // 验证结果
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0], "1");
}
