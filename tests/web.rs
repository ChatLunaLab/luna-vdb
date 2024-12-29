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
            embeddings: vec![0.8, 0.7, 0.6, 0.2, 0.1], // 猫的向量
        },
        EmbeddedResource {
            id: "dog".to_string(),
            embeddings: vec![0.7, 0.8, 0.6, 0.3, 0.1], // 狗的向量 (与猫相似)
        },
        EmbeddedResource {
            id: "bird".to_string(),
            embeddings: vec![0.6, 0.5, 0.8, 0.4, 0.2], // 鸟的向量 (与猫狗稍有不同)
        },
        EmbeddedResource {
            id: "fish".to_string(),
            embeddings: vec![0.2, 0.3, 0.4, 0.8, 0.7], // 鱼的向量 (差异较大)
        },
        EmbeddedResource {
            id: "car".to_string(),
            embeddings: vec![-0.1, -0.2, -0.3, -0.8, -0.9], // 汽车的向量 (完全不同)
        },
    ];

    let resource = Resource { embeddings };
    luna_vdb.index(resource);

    // 测试场景1: 搜索最接近"猫"的向量
    console_log!("Testing cat-like vector search");
    let cat_query = vec![0.8, 0.7, 0.6, 0.2, 0.1];
    let neighbors = luna_vdb.search(cat_query, 3);
    assert_eq!(neighbors.len(), 3);
    assert_eq!(neighbors[0], "cat"); // 第一个应该是猫
    assert_eq!(neighbors[1], "dog"); // 第二个应该是狗（因为向量最相似）
    assert_eq!(neighbors[2], "bird"); // 第三个应该是鸟

    // 测试场景2: 搜索介于猫狗之间的向量
    console_log!("Testing intermediate cat-dog vector search");
    let cat_dog_query = vec![0.75, 0.75, 0.6, 0.25, 0.1];
    let neighbors = luna_vdb.search(cat_dog_query, 2);
    assert_eq!(neighbors.len(), 2);
    // 猫狗应该都在结果中，顺序可能略有不同
    assert!(neighbors.contains(&"cat".to_string()));
    assert!(neighbors.contains(&"dog".to_string()));

    // 测试场景3: 搜索完全不同类别的向量
    console_log!("Testing vehicle-like vector search");
    let car_query = vec![-0.15, -0.25, -0.35, -0.85, -0.95];
    let neighbors = luna_vdb.search(car_query, 1);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0], "car");

    // 测试场景4: 搜索水生动物相关的向量
    console_log!("Testing aquatic-like vector search");
    let aquatic_query = vec![0.2, 0.3, 0.4, 0.85, 0.75];
    let neighbors = luna_vdb.search(aquatic_query, 2);
    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0], "fish"); // 第一个应该是鱼
                                      // 第二个结果应该与鱼的相似度明显较低

    // 测试场景5: 边界情况测试
    console_log!("Testing boundary case search");
    // 测试一个与所有向量都有一定距离的查询向量
    let boundary_query = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let neighbors = luna_vdb.search(boundary_query, 5);
    assert_eq!(neighbors.len(), 5); // 应该返回所有向量

    // 验证返回的是所有向量
    let result_set: std::collections::HashSet<_> = neighbors.into_iter().collect();
    assert_eq!(result_set.len(), 5);
    assert!(result_set.contains("cat"));
    assert!(result_set.contains("dog"));
    assert!(result_set.contains("bird"));
    assert!(result_set.contains("fish"));
    assert!(result_set.contains("car"));
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
    assert!(luna_vdb.remove(ids));
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
    assert_eq!(neighbors.len(), 10);

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
        assert_eq!(results.len(), 4);
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
    assert_eq!(luna_vdb.size(), (400 - remove_count) as u64);

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
    assert_eq!(results.len(), 20);
}
