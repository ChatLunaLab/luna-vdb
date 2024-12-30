//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
extern crate web_sys;

use luna_vdb::*;
use wasm_bindgen_test::*;

// 定义 console_log 宏
macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format!($($t)*).into()))
}

wasm_bindgen_test_configure!(run_in_browser);


fn random_number() -> i32 {
    use getrandom::getrandom;

    let mut buffer = [0u8; 4];

    // 从系统熵池中获取随机数
    getrandom(&mut buffer).expect("Failed to generate random bytes");

    // 将字节数组转换为整数
    return i32::from_le_bytes(buffer);
}

fn random_string(length: usize) -> String {
    // ascii
    let mut s = String::with_capacity(length);
    for _ in 0..length {
        s.push(char::from_u32((random_number() % 26 + 97).try_into().unwrap()).unwrap());
    }
    s
}

fn generate_test_data(count: usize, dim: usize) -> Vec<EmbeddedResource> {
    use getrandom::getrandom;

    let mut resources = Vec::with_capacity(count);
    for _ in 0..count {
        let mut embeddings = Vec::with_capacity(dim);
        for _ in 0..dim {
            // 生成[-1, 1]范围内的随机浮点数
            let mut buffer = [0u8; 4];
            getrandom(&mut buffer).expect("Failed to generate random bytes");
            let value = random_number() as f32 / i32::MAX as f32 * 2.0 - 1.0;
            embeddings.push(value);
        }
        resources.push(EmbeddedResource {
            id: random_string(10),
            embeddings,
        });
    }
    resources
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

    // 创建一组相似度不同的文本向量
    let embeddings = vec![
        EmbeddedResource {
            id: "cat".to_string(),
            embeddings: vec![0.8, 0.7, 0.6, 0.2, 0.1], 
        },
        EmbeddedResource {
            id: "dog".to_string(),
            embeddings: vec![0.7, 0.8, 0.6, 0.3, 0.1], 
        },
        EmbeddedResource {
            id: "bird".to_string(),
            embeddings: vec![0.6, 0.5, 0.8, 0.4, 0.2], 
        },
        EmbeddedResource {
            id: "fish".to_string(),
            embeddings: vec![0.2, 0.3, 0.4, 0.8, 0.7], 
        },
        EmbeddedResource {
            id: "car".to_string(),
            embeddings: vec![-0.1, -0.2, -0.3, -0.8, -0.9], 
        },
    ];

    let resource = Resource { embeddings };
    luna_vdb.index(resource);

    // 测试场景1: 搜索最接近"猫"的向量
    console_log!("Testing cat-like vector search");
    let cat_query = vec![0.8, 0.7, 0.6, 0.2, 0.1];
    let result = luna_vdb.search(cat_query, 3);
    assert_eq!(result.neighbors.len(), 3);
    assert_eq!(result.neighbors[0].id, "cat");
    assert_eq!(result.neighbors[1].id, "dog");
    assert_eq!(result.neighbors[2].id, "bird");
    
    // 验证距离值是递增的
    assert!(result.neighbors[0].distance < result.neighbors[1].distance);
    assert!(result.neighbors[1].distance < result.neighbors[2].distance);

    // 测试场景2: 搜索边界值向量
    console_log!("Testing boundary vector search");
    let boundary_query = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let result = luna_vdb.search(boundary_query, 5);
    assert_eq!(result.neighbors.len(), 5);
    
    // 验证所有结果都有合理的距离值
    for neighbor in &result.neighbors {
        assert!(neighbor.distance >= 0.0);
    }

    // 测试场景3: 搜索零向量
    console_log!("Testing zero vector search");
    let zero_query = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let result = luna_vdb.search(zero_query, 3);
    assert_eq!(result.neighbors.len(), 3);

    // 测试场景4: 搜索负向量
    console_log!("Testing negative vector search");
    let negative_query = vec![-0.1, -0.2, -0.3, -0.8, -0.9];
    let result = luna_vdb.search(negative_query, 1);
    assert_eq!(result.neighbors[0].id, "car");
    assert!(result.neighbors[0].distance < 0.1); // 应该非常接近

    // 测试场景5: 验证距离计算
    console_log!("Testing distance calculations");
    let query = vec![0.8, 0.7, 0.6, 0.2, 0.1];  // 与 cat 向量相同
    let result = luna_vdb.search(query, 1);
    assert_eq!(result.neighbors[0].id, "cat");
    assert!(result.neighbors[0].distance < 1e-6); // 应该几乎为0

    // 测试场景6: 极限搜索数量
    console_log!("Testing search with max k");
    let result = luna_vdb.search(vec![0.0; 5], 10);
    assert_eq!(result.neighbors.len(), 5); // 不应超过实际存在的向量数量
}

#[wasm_bindgen_test]
fn test_luna_vdb_add_remove() {
   
    console_log!("Starting test_luna_vdb_add_remove");
    
    let mut luna_vdb = LunaVDB::new(None);

    // 测试添加
    let embeddings = vec![EmbeddedResource {
        id: "3".to_string(),
        embeddings: vec![0.7, 0.8, 0.9],
    }];
    let resource = Resource { embeddings };
    luna_vdb.add(resource);
    assert_eq!(luna_vdb.size(), 1);

    // 测试移除 - 使用新的方式
    let ids = vec!["3".to_string()];
    luna_vdb.remove(ids);
    assert_eq!(luna_vdb.size(), 0);


}

#[wasm_bindgen_test]
fn test_luna_vdb_serialization() {
   
    console_log!("Starting test_luna_vdb_serialization");
    
    let mut luna_vdb = LunaVDB::new(None);

    // 添加一些数据
    let embeddings = vec![EmbeddedResource {
        id: "1".to_string(),
        embeddings: vec![0.1, 0.2, 0.3],
    }];
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
    let embeddings = vec![EmbeddedResource {
        id: "1".to_string(),
        embeddings: vec![0.1, 0.2, 0.3],
    }];
    let resource = Resource { embeddings };
    luna_vdb.index(resource);
    assert_eq!(luna_vdb.size(), 1);

    // 测试清空
    luna_vdb.clear();
    assert_eq!(luna_vdb.size(), 0);

   
}

#[wasm_bindgen_test]
fn test_luna_vdb_large_dataset() {
   
    console_log!("Starting test_luna_vdb_large_dataset");
    
    let mut luna_vdb = LunaVDB::new(None);

    // 生成1000个128维的测试向量
    let embeddings = generate_test_data(1000, 1024);
    let resource = Resource { embeddings };

    // 测试大规模索引
    console_log!("Indexing 1000 vectors...");
    luna_vdb.index(resource);
    assert_eq!(luna_vdb.size(), 1000);

    // 测试批量搜索
    console_log!("Testing batch search...");
    let query = vec![0.5; 1024]; // 创建一个1024维的查询向量
    let neighbors = luna_vdb.search(query, 10);
    assert_eq!(neighbors.neighbors.len(), 10);

    // 测试增量更新
    console_log!("Testing incremental updates...");
    let new_embeddings = generate_test_data(100, 1024);
    let new_resource = Resource {
        embeddings: new_embeddings,
    };
    luna_vdb.add(new_resource);
    assert_eq!(luna_vdb.size(), 1100);


}

#[wasm_bindgen_test]
fn test_luna_vdb_edge_cases() {
    console_log!("Starting test_luna_vdb_edge_cases");
    
    let mut luna_vdb = LunaVDB::new(None);

    // 测试极端值
    let embeddings = vec![
        EmbeddedResource {
            id: "max".to_string(),
            embeddings: vec![f32::MAX; 10],
        },
        EmbeddedResource {
            id: "min".to_string(),
            embeddings: vec![f32::MIN; 10],
        },
        EmbeddedResource {
            id: "zero".to_string(),
            embeddings: vec![0.0; 10],
        },
        EmbeddedResource {
            id: "mixed".to_string(),
            embeddings: vec![
                1.0,
                -1.0,
                f32::MAX,
                f32::MIN,
                0.0,
                0.5,
                -0.5,
                f32::EPSILON,
                -f32::EPSILON,
                1.0,
            ],
        },
    ];

    let resource = Resource { embeddings };
    luna_vdb.index(resource);

    // 使用不同类型的查询向量测试
    let queries = vec![
        vec![0.0; 10],  // 零向量
        vec![1.0; 10],  // 单位向量
        vec![-1.0; 10], // 负单位向量
    ];

    for (i, query) in queries.iter().enumerate() {
        console_log!("Testing query type {}", i);
        let results = luna_vdb.search(query.clone(), 4);
        assert_eq!(results.neighbors.len(), 4);
    }


}

#[wasm_bindgen_test]
fn test_luna_vdb_persistence() {
   
    console_log!("Starting test_luna_vdb_persistence");
    
    let mut luna_vdb = LunaVDB::new(None);

    // 生成大量测试数据
    let initial_embeddings = generate_test_data(500, 1024);
    let test_queries = initial_embeddings[0..5].to_vec();
    let resource = Resource {
        embeddings: initial_embeddings,
    };
    luna_vdb.index(resource);

    // 序列化
    let serialized = luna_vdb.serialize();

    // 创建新实例并反序列化
    let new_luna_vdb = LunaVDB::deserialize(serialized);

    // 验证数据完整性
    assert_eq!(new_luna_vdb.size(), 500);

    // 在新实例上进行搜索测试
    for (i, query) in test_queries.iter().enumerate() {
        console_log!("Testing query {} on restored database", i);
        let original_results = luna_vdb.search(query.embeddings.clone(), 5);
        let new_results = new_luna_vdb.search(query.embeddings.clone(), 5);
        assert_eq!(original_results, new_results);
    }

}

#[wasm_bindgen_test]
fn test_luna_vdb_dynamic_operations() {
   
    console_log!("Starting test_luna_vdb_dynamic_operations");
    
    let mut luna_vdb = LunaVDB::new(None);

    // 初始数据
    let mut all_ids = Vec::new();
    let initial_embeddings = generate_test_data(400, 32);
    for resource in &initial_embeddings {
        all_ids.push(resource.id.clone());
    }

    luna_vdb.index(Resource {
        embeddings: initial_embeddings,
    });

    // 随机删除一些向量
    let remove_count = 50;
    let mut to_remove = Vec::new();
    for i in 0..remove_count {
        let idx = i * 4; // 间隔删除
        to_remove.push(all_ids[idx].clone());
    }

    console_log!("Removing {} vectors", remove_count);
    luna_vdb.remove(to_remove);
    assert_eq!(luna_vdb.size(), 400 - remove_count);

    // 添加新的向量
    let new_embeddings = generate_test_data(100, 32);
    console_log!("Adding 100 new vectors");
    luna_vdb.add(Resource {
        embeddings: new_embeddings,
    });
    assert_eq!(luna_vdb.size(), 450);

    // 执行复杂搜索
    let complex_query = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        .into_iter()
        .cycle()
        .take(32)
        .collect::<Vec<f32>>();

    let results = luna_vdb.search(complex_query, 20);
    assert_eq!(results.neighbors.len(), 20);

}
