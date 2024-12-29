extern crate wasm_bindgen_test;
use luna_vdb::*;
use wasm_bindgen_test::*;

// 定义 console_log 宏 (Node.js 环境)
macro_rules! console_log {
    ($($t:tt)*) => (println!($($t)*))
}


#[wasm_bindgen_test]
fn test_luna_vdb_basic() {
    console_log!("Starting test_luna_vdb_basic");
    let mut luna_vdb = LunaVDB::new(None);
    assert_eq!(luna_vdb.size(), 0);
    
    // 测试初始索引
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
    luna_vdb.index(resource);
    assert_eq!(luna_vdb.size(), 2);
}

#[wasm_bindgen_test]
fn test_luna_vdb_search() {
    console_log!("Starting test_luna_vdb_search");
    let mut luna_vdb = LunaVDB::new(None);
    
    // 创建测试数据
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
    luna_vdb.index(resource);

    // 测试搜索
    let query = vec![0.15, 0.25, 0.35];
    let neighbors = luna_vdb.search(query.clone(), 1);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0], "1");

    // 测试搜索多个结果
    let neighbors = luna_vdb.search(query, 2);
    assert_eq!(neighbors.len(), 2);
}

#[wasm_bindgen_test]
fn test_luna_vdb_add_remove() {
    console_log!("Starting test_luna_vdb_add_remove");
    let mut luna_vdb = LunaVDB::new(None);
    
    // 测试添加
    let embeddings = vec![
        EmbeddedResource {
            id: "3".to_string(),
            embeddings: vec![0.7, 0.8, 0.9],
        },
    ];
    let resource = Resource { embeddings };
    luna_vdb.add(resource);
    assert_eq!(luna_vdb.size(), 1);

    // 测试移除
    let embeddings = vec![
        EmbeddedResource {
            id: "3".to_string(),
            embeddings: vec![0.7, 0.8, 0.9],
        },
    ];
    let resource = Resource { embeddings };
    assert!(luna_vdb.remove(resource));
    assert_eq!(luna_vdb.size(), 0);
}

#[wasm_bindgen_test]
fn test_luna_vdb_serialization() {
    console_log!("Starting test_luna_vdb_serialization");
    let mut luna_vdb = LunaVDB::new(None);
    
    // 添加一些数据
    let embeddings = vec![
        EmbeddedResource {
            id: "1".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
        },
    ];
    let resource = Resource { embeddings };
    luna_vdb.index(resource);

    // 测试序列化
    let serialized = luna_vdb.serialize();
    assert!(!serialized.is_empty());

    // 测试反序列化
    let new_luna_vdb = LunaVDB::deserialize(serialized);
    assert_eq!(new_luna_vdb.size(), 1);

    // 验证搜索结果一致性
    let query = vec![0.15, 0.25, 0.35];
    let original_results = luna_vdb.search(query.clone(), 1);
    let new_results = new_luna_vdb.search(query, 1);
    assert_eq!(original_results, new_results);
}

#[wasm_bindgen_test]
fn test_luna_vdb_clear() {
    console_log!("Starting test_luna_vdb_clear");
    let mut luna_vdb = LunaVDB::new(None);
    
    // 添加数据
    let embeddings = vec![
        EmbeddedResource {
            id: "1".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
        },
    ];
    let resource = Resource { embeddings };
    luna_vdb.index(resource);
    assert_eq!(luna_vdb.size(), 1);

    // 测试清空
    luna_vdb.clear();
    assert_eq!(luna_vdb.size(), 0);
}
