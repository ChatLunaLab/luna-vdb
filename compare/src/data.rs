//! Synthetic corpora shared by the `compare` and `sweep` binaries.

pub fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(splitmix(seed) | 1)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal, Box–Muller.
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(f32::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy)]
pub enum Data {
    /// Uniform in `[-0.5, 0.5)^dim`: no structure at all. The worst case for
    /// any partitioning index.
    Uniform,
    /// Gaussian mixture: `n / 64` centres, uniform in the cube, each point a
    /// centre plus isotropic noise. The noise is set so clusters overlap
    /// noticeably — a point's 10 nearest neighbours are usually, but not
    /// always, in its own cluster — which is closer to real text embeddings
    /// than either well-separated blobs or uniform noise.
    Clustered,
}

impl Data {
    pub fn name(self) -> &'static str {
        match self {
            Data::Uniform => "uniform",
            Data::Clustered => "clustered",
        }
    }
}

/// Corpus and queries drawn from the same distribution, so a query is a new
/// point "about" one of the corpus's topics, not a copy of a stored row.
pub fn generate(
    data: Data,
    count: usize,
    queries: usize,
    dim: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Rng::new(seed);
    match data {
        Data::Uniform => {
            let mut draw = |n: usize| -> Vec<Vec<f32>> {
                (0..n)
                    .map(|_| (0..dim).map(|_| rng.unit() - 0.5).collect())
                    .collect()
            };
            let corpus = draw(count);
            let queries = draw(queries);
            (corpus, queries)
        }
        Data::Clustered => {
            let centres_n = (count / 64).max(1);
            let centres: Vec<Vec<f32>> = (0..centres_n)
                .map(|_| (0..dim).map(|_| rng.unit() - 0.5).collect())
                .collect();
            // Per-coordinate spread of the centres is 1/sqrt(12) ≈ 0.289; noise
            // at 0.6 of that makes within-cluster distances comparable to, but
            // smaller than, the gap to neighbouring clusters.
            let sigma = 0.6 * 0.2887;
            let mut draw = |n: usize| -> Vec<Vec<f32>> {
                (0..n)
                    .map(|_| {
                        let pick = ((rng.unit() * centres_n as f32) as usize).min(centres_n - 1);
                        centres[pick]
                            .iter()
                            .map(|c| c + sigma * rng.normal())
                            .collect()
                    })
                    .collect()
            };
            let corpus = draw(count);
            let queries = draw(queries);
            (corpus, queries)
        }
    }
}
