// 本地 Rust 测试，不依赖 WASM
use luna_vdb::{EmbeddedResource, LunaVDB, Resource};

fn random_number() -> i32 {
    use getrandom::getrandom;

    let mut buffer = [0u8; 4];
    getrandom(&mut buffer).expect("Failed to generate random bytes");
    i32::from_le_bytes(buffer)
}

fn random_string(length: usize) -> String {
    let mut s = String::with_capacity(length);
    for _ in 0..length {
        s.push(char::from_u32((random_number().abs() % 26 + 97).try_into().unwrap()).unwrap());
    }
    s
}

fn generate_test_data(count: usize, dim: usize) -> Vec<EmbeddedResource> {
    let mut resources = Vec::with_capacity(count);
    for _ in 0..count {
        let mut embeddings = Vec::with_capacity(dim);
        for _ in 0..dim {
            let value = random_number() as f32 / i32::MAX as f32;
            embeddings.push(value);
        }
        resources.push(EmbeddedResource {
            id: random_string(10),
            embeddings,
        });
    }
    resources
}

#[test]
fn test_engine_basic() {
    let mut engine = LunaVDB::new(None);
    assert_eq!(engine.size(), 0);

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
    engine.index(resource);
    assert_eq!(engine.size(), 2);
}

#[test]
fn test_engine_search() {
    let mut engine = LunaVDB::new(None);

    // 创建一组相似度不同的向量
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
    engine.index(resource);

    // 测试场景1: 搜索最接近"猫"的向量
    let cat_query = vec![0.8, 0.7, 0.6, 0.2, 0.1];
    let result = engine.search(cat_query, 3);
    assert_eq!(result.neighbors.len(), 3);
    assert_eq!(result.neighbors[0].id, "cat");
    assert_eq!(result.neighbors[1].id, "dog");
    assert_eq!(result.neighbors[2].id, "bird");

    // 验证距离值是递增的
    assert!(result.neighbors[0].distance < result.neighbors[1].distance);
    assert!(result.neighbors[1].distance < result.neighbors[2].distance);

    // 测试场景2: 搜索边界值向量
    let boundary_query = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let result = engine.search(boundary_query, 5);
    assert_eq!(result.neighbors.len(), 5);

    // 验证所有结果都有合理的距离值
    for neighbor in &result.neighbors {
        assert!(neighbor.distance >= 0.0);
    }

    // 测试场景3: 搜索零向量
    let zero_query = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let result = engine.search(zero_query, 3);
    assert_eq!(result.neighbors.len(), 3);

    // 测试场景4: 搜索负向量
    let negative_query = vec![-0.1, -0.2, -0.3, -0.8, -0.9];
    let result = engine.search(negative_query, 1);
    assert_eq!(result.neighbors[0].id, "car");
    assert!(result.neighbors[0].distance < 0.1);

    // 测试场景5: 验证距离计算
    let query = vec![0.8, 0.7, 0.6, 0.2, 0.1]; // 与 cat 向量相同
    let result = engine.search(query, 1);
    assert_eq!(result.neighbors[0].id, "cat");
    assert!(result.neighbors[0].distance < 1e-6);

    // 测试场景6: 极限搜索数量
    let result = engine.search(vec![0.0; 5], 10);
    assert_eq!(result.neighbors.len(), 5); // 不应超过实际存在的向量数量
}

#[test]
fn test_engine_add_remove() {
    let mut engine = LunaVDB::new(None);

    // 测试添加
    let embeddings = vec![EmbeddedResource {
        id: "3".to_string(),
        embeddings: vec![0.7, 0.8, 0.9],
    }];
    let resource = Resource { embeddings };
    engine.add(resource);
    assert_eq!(engine.size(), 1);

    // 测试移除
    let ids = vec!["3".to_string()];
    engine.remove(ids);
    assert_eq!(engine.size(), 0);
}

#[test]
fn test_engine_serialization() {
    let mut engine = LunaVDB::new(None);

    // 添加一些数据
    let embeddings = vec![EmbeddedResource {
        id: "1".to_string(),
        embeddings: vec![0.1, 0.2, 0.3],
    }];
    let resource = Resource { embeddings };
    engine.index(resource);

    // 测试序列化
    let serialized = engine.serialize();
    assert!(!serialized.is_empty());

    // 测试反序列化
    let new_engine = LunaVDB::deserialize(serialized);
    assert_eq!(new_engine.size(), 1);

    // 验证搜索结果一致性
    let query = vec![0.15, 0.25, 0.35];
    let original_results = engine.search(query.to_owned(), 1);
    let new_results = new_engine.search(query.to_owned(), 1);
    assert_eq!(
        original_results.neighbors.len(),
        new_results.neighbors.len()
    );
    assert_eq!(
        original_results.neighbors[0].id,
        new_results.neighbors[0].id
    );
}

#[test]
fn test_engine_clear() {
    let mut engine = LunaVDB::new(None);

    // 添加数据
    let embeddings = vec![EmbeddedResource {
        id: "1".to_string(),
        embeddings: vec![0.1, 0.2, 0.3],
    }];
    let resource = Resource { embeddings };
    engine.index(resource);
    assert_eq!(engine.size(), 1);

    // 测试清空
    engine.clear();
    assert_eq!(engine.size(), 0);
}

#[test]
fn test_engine_large_dataset() {
    let mut engine = LunaVDB::new(None);

    // 生成1000个1024维的测试向量
    let embeddings = generate_test_data(1000, 9024);
    let resource = Resource { embeddings };

    // 测试大规模索引
    engine.index(resource);
    assert_eq!(engine.size(), 1000);

    // 测试批量搜索
    let query = vec![0.5; 1024];
    let neighbors = engine.search(query, 10);
    assert_eq!(neighbors.neighbors.len(), 10);

    // 测试增量更新
    let new_embeddings = generate_test_data(100, 9024);
    let new_resource = Resource {
        embeddings: new_embeddings,
    };
    engine.add(new_resource);
    assert_eq!(engine.size(), 1100);
}

#[test]
fn test_engine_edge_cases() {
    let mut engine = LunaVDB::new(None);

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
    engine.index(resource);

    // 使用不同类型的查询向量测试
    let queries = vec![
        vec![0.0; 10],  // 零向量
        vec![1.0; 10],  // 单位向量
        vec![-1.0; 10], // 负单位向量
    ];

    for query in queries.iter() {
        let results = engine.search(query.to_vec(), 4);
        assert_eq!(results.neighbors.len(), 4);
    }
}

#[test]
fn test_engine_persistence() {
    let mut engine = LunaVDB::new(None);

    // 生成测试数据
    let initial_embeddings = generate_test_data(500, 1024);
    let test_queries = initial_embeddings[0..5].to_vec();
    let resource = Resource {
        embeddings: initial_embeddings,
    };
    engine.index(resource);

    // 序列化
    let serialized = engine.serialize();

    // 创建新实例并反序列化
    let new_engine = LunaVDB::deserialize(serialized);

    // 验证数据完整性
    assert_eq!(new_engine.size(), 500);

    // 在新实例上进行搜索测试
    for query in test_queries.iter() {
        let original_results = engine.search(query.embeddings.to_owned(), 5);
        let new_results = new_engine.search(query.embeddings.to_owned(), 5);
        assert_eq!(
            original_results.neighbors.len(),
            new_results.neighbors.len()
        );
        for (orig, new) in original_results
            .neighbors
            .iter()
            .zip(new_results.neighbors.iter())
        {
            assert_eq!(orig.id, new.id);
        }
    }
}

#[test]
fn test_engine_dynamic_operations() {
    let mut engine = LunaVDB::new(None);

    // 初始数据
    let mut all_ids = Vec::new();
    let initial_embeddings = generate_test_data(400, 32);
    for resource in &initial_embeddings {
        all_ids.push(resource.id.clone());
    }

    engine.index(Resource {
        embeddings: initial_embeddings,
    });

    // 随机删除一些向量
    let remove_count = 50;
    let mut to_remove = Vec::new();
    for i in 0..remove_count {
        let idx = i * 4; // 间隔删除
        to_remove.push(all_ids[idx].clone());
    }

    engine.remove(to_remove);
    assert_eq!(engine.size(), 400 - remove_count);

    // 添加新的向量
    let new_embeddings = generate_test_data(100, 32);
    engine.add(Resource {
        embeddings: new_embeddings,
    });
    assert_eq!(engine.size(), 450);

    // 执行复杂搜索
    let complex_query = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
        .into_iter()
        .cycle()
        .take(32)
        .collect::<Vec<f32>>();

    let results = engine.search(complex_query, 20);
    assert_eq!(results.neighbors.len(), 20);
}

#[test]
fn test_engine_empty_search() {
    let engine = LunaVDB::new(None);

    // 在空引擎上搜索
    let query = vec![0.1, 0.2, 0.3];
    let result = engine.search(query, 5);
    assert_eq!(result.neighbors.len(), 0);
}

#[test]
fn test_engine_single_vector() {
    let mut engine = LunaVDB::new(None);

    // 添加单个向量
    let embeddings = vec![EmbeddedResource {
        id: "single".to_string(),
        embeddings: vec![1.0, 2.0, 3.0],
    }];
    engine.index(Resource { embeddings });

    // 搜索
    let query = vec![1.0, 2.0, 3.0];
    let result = engine.search(query, 1);
    assert_eq!(result.neighbors.len(), 1);
    assert_eq!(result.neighbors[0].id, "single");
    assert!(result.neighbors[0].distance < 1e-6);
}

#[test]
fn test_engine_duplicate_ids() {
    let mut engine = LunaVDB::new(None);

    // 添加相同 ID 的向量
    let embeddings = vec![
        EmbeddedResource {
            id: "duplicate".to_string(),
            embeddings: vec![0.1, 0.2, 0.3],
        },
        EmbeddedResource {
            id: "duplicate".to_string(),
            embeddings: vec![0.4, 0.5, 0.6],
        },
    ];
    engine.index(Resource { embeddings });

    // 引擎应该能够处理重复 ID
    assert!(engine.size() > 0);
}
