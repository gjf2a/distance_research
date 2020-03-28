use crate::mnist_data::{Image, ImageIterator, Grid};
use rand_distr::{Normal, Distribution};
use rand::prelude::ThreadRng;
use crate::bits::BitArray;
use rand::distributions::Uniform;
use crate::hash_histogram::HashHistogram;

#[derive(Clone)]
pub struct Descriptor {
    pairs: Vec<((usize,usize),(usize,usize))>,
    width: usize,
    height: usize
}

fn constrained_random(dist: &Normal<f64>, rng: &mut ThreadRng, max: usize) -> usize {
    let mut value = dist.sample(rng);
    value = value.max(0 as f64);
    value = value.min((max - 1) as f64);
    value as usize
}

impl Descriptor {
    pub fn classic_gaussian_brief(n: usize, width: usize, height: usize) -> Descriptor {
        let mut rng = rand::thread_rng();
        let x_dist = Normal::new((width/2) as f64, (width/6) as f64).unwrap();
        let y_dist = Normal::new((height/2) as f64, (height/6) as f64).unwrap();
        let mut result = Descriptor {pairs: Vec::new(), width, height};
        for _ in 0..n {
            result.pairs.push(((constrained_random(&x_dist, &mut rng, width),
                                constrained_random(&y_dist, &mut rng, height)),
                              (constrained_random(&x_dist, &mut rng, width),
                                constrained_random(&y_dist, &mut rng, height))));
        }
        result
    }

    pub fn classic_uniform_brief(n: usize, width: usize, height: usize) -> Descriptor {
        let mut rng = rand::thread_rng();
        let x_dist = Uniform::new(0, width);
        let y_dist = Uniform::new(0, height);
        let mut result = Descriptor {pairs: Vec::new(), width, height};
        for _ in 0..n {
            result.pairs.push(((x_dist.sample(&mut rng), y_dist.sample(&mut rng)),
                              (x_dist.sample(&mut rng), y_dist.sample(&mut rng))));
        }
        result
    }

    pub fn uniform_neighbor(neighbors: usize, width: usize, height: usize) -> Descriptor {
        let mut rng = rand::thread_rng();
        let x_dist = Uniform::new(0, width);
        let y_dist = Uniform::new(0, height);
        let mut result = Descriptor {pairs: Vec::new(), width, height};
        ImageIterator::new(0, 0, width, height, 1)
            .for_each(|(x, y)|
                for _ in 0..neighbors {
                    result.pairs.push(((x, y), (x_dist.sample(&mut rng), y_dist.sample(&mut rng))));
                });
        result
    }

    pub fn gaussian_neighbor(neighbors: usize, stdev: usize, width: usize, height: usize) -> Descriptor {
        let x_dist = Normal::new(0 as f64, stdev as f64).unwrap();
        let y_dist = Normal::new(0 as f64, stdev as f64).unwrap();
        let mut result = Descriptor {pairs: Vec::new(), width, height};
        ImageIterator::new(0, 0, width, height, 1)
            .for_each(|(x, y)|
                for _ in 0..neighbors {
                    let x_other = random_bounded_normal_value(&x_dist, x, 0, width);
                    let y_other = random_bounded_normal_value(&y_dist, y, 0, height);
                    assert!(x_other < width);
                    assert!(y_other < height);
                    result.pairs.push(((x, y), (x_other, y_other)));
                });
        result
    }

    pub fn equidistant(width: usize, height: usize, x_offset: usize, y_offset: usize) -> Descriptor {
        let mut result = Descriptor {pairs: Vec::new(), width, height};
        ImageIterator::new(0, 0, width, height, 1)
            .for_each(|(x, y)|
                result.pairs.push(((x, y), ((x + x_offset) % width, ((y + y_offset) % height))))
            );
        result
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn apply_to(&self, img: &Image) -> BitArray {
        assert_eq!(img.side(), self.width());
        assert_eq!(img.side(), self.height());

        let mut bits = BitArray::new();
        self.pairs.iter()
            .for_each(|((x1, y1), (x2, y2))|
                bits.add(img.get(*x1, *y1) < img.get(*x2, *y2)));
        bits
    }

    pub fn evaluate(&self, img: &Image, x1: usize, y1: usize, x2: usize, y2: usize) -> bool {
        img.get(x1, y1) < img.get(x2, y2)
    }

    pub fn majority_image(&self, img: &Image) -> BitArray {
        let mut counts = HashHistogram::new();
        self.pairs.iter()
            .for_each(|((x1, y1), (x2, y2))|
                counts.bump((x1, y1, img.get(*x1, *y1) < img.get(*x2, *y2))));
        let mut bits = BitArray::new();
        img.x_y_iter()
            .for_each(|(x, y)|
                bits.add(counts.get((&x, &y, true)) > counts.get((&x, &y, false))));
        bits
    }
}

pub fn random_bounded_normal_value(dist: &Normal<f64>, start_value: usize, min: usize, max: usize) -> usize {
    let mut rng = ThreadRng::default();
    let sample = dist.sample(&mut rng).abs() as usize;
    let min_diff = start_value - min;
    let max_diff = max - start_value;

    if sample < min_diff && sample < max_diff {
        if rand::random() {
            start_value + sample
        } else {
            start_value - sample
        }
    } else if sample < min_diff {
        start_value - sample
    } else if sample < max_diff {
        start_value + sample
    } else if rand::random() {
        min
    } else {
        max - 1
    }
}