use crate::mnist_data::{Image, Grid};
use crate::kernel_patch::kernelize_single_image;
use crate::euclidean_distance::euclidean_distance;

pub fn find_keypoints(img: &Image, num_kernels: usize, kernel_size: usize, num_keypoints: usize) -> Vec<(usize, usize)> {
    let kernels = kernelize_single_image(img, num_kernels, kernel_size);
    let mut dists: Vec<(u32, usize, usize)> = img.x_y_step_iter(1)
        .map(|(x, y)| (best_matching_kernel_distance(&kernels, img, x, y), x, y))
        .collect();
    dists.sort();
    dists.iter()
        .take(num_keypoints)
        .map(|(_, x, y)| (*x, *y))
        .collect()
}

pub fn closest_for_all(group1: &Vec<(usize,usize)>, group2: &Vec<(usize,usize)>) -> usize {
    closest_for_all_one_way(group1, group2) + closest_for_all_one_way(group2, group1)
}

pub fn closest_for_all_one_way(group1: &Vec<(usize,usize)>, group2: &Vec<(usize,usize)>) -> usize {
    group1.iter()
        .map(|p| best_matching_distance(*p, group2))
        .sum()
}

pub fn best_matching_distance(candidate: (usize,usize), references: &Vec<(usize,usize)>) -> usize {
    references.iter()
        .map(|(x,y)| squared_diff(*x, candidate.0) + squared_diff(*y, candidate.1))
        .min()
        .unwrap()
}

pub fn squared_diff(x1: usize, x2: usize) -> usize {
    let (min, max) = if x1 < x2 {(x1, x2)} else {(x2, x1)};
    (max - min).pow(2)
}

pub fn best_matching_kernel_distance(kernels: &Vec<Image>, img: &Image, x: usize, y: usize) -> u32 {
    (0..kernels.len())
        .map(|i| euclidean_distance(&img.subimage(x, y, kernels[i].side()), &kernels[i]))
        .min()
        .unwrap()
}