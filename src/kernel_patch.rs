use crate::mnist_data::{Image, image_mean};
use crate::euclidean_distance::euclidean_distance;
use crate::convolutional::add_kernels_from_to;

pub fn kernelize_single_image(img: &Image, num_kernels: usize, kernel_size: usize) -> Vec<Image> {
    let mut candidates = Vec::new();
    add_kernels_from_to(img, &mut candidates, kernel_size);
    kmeans::Kmeans::new(num_kernels, &candidates, euclidean_distance, image_mean).move_means()
}

pub fn best_match_distance(k1: &Vec<Image>, k2: &Vec<Image>) -> u32 {
    assert_eq!(k1.len(), k2.len());
    best_match_one_way(k1, k2) + best_match_one_way(k2, k1)
}

fn best_match_one_way(k1: &Vec<Image>, k2: &Vec<Image>) -> u32 {
    assert_eq!(k1.len(), k2.len());
    assert!(k2.len() > 0);
    k1.iter()
        .map(|img1| k2.iter()
            .map(|img2| euclidean_distance(img1, img2))
            .min()
            .unwrap())
        .sum()
}