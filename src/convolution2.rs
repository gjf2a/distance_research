// Algorithm idea:
//
// Use 3x3 kernels.
//
// Use kmeans++ to mine kernels from all of the input images.
//
// Use a stride of 2 (alternating pixels). This way, no pixel appears in any other pixel's window.
// - In fact, a stride of 3 might be even better, so that the windows don't overlap.
//
// At the next stage, each pixel is replaced by an integer referencing which kernel was its
// closest match.
//
// Having applied all of these kernels to all input images, we get a set of level 2 images.
// We cluster again to create a new set of kernels for this level. For this round, we use
// Hamming distance and some kind of max (mode) for a mean.
//
// Keep doing this until images get super small. We'll have to do research to figure out how
// small is too small. Probably just try out each level incrementally until utility tails off.
//
// From here, we can use knn on the final output images for each training sample.
//
// We should be able to reconstruct a representation of the original image by projecting
// backward through each level, expanding each pixel with its best matching 3x3 kernel.
// **Updated Idea** Just keep all the images from the previous levels.

use crate::mnist_data::{Image, image_mean, Grid};
use crate::convolutional::{extract_kernels_from, add_kernels_from_to};
use crate::euclidean_distance::euclidean_distance;
use hash_histogram::mode;
use std::cmp::Ordering;

const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;

#[derive(Copy, Clone, PartialEq)]
struct KernelPyramidDistance {
    num_levels_identical: usize,
    distance_level_n: u32
}

impl PartialOrd for KernelPyramidDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.num_levels_identical == other.num_levels_identical {
            self.distance_level_n.partial_cmp(&other.distance_level_n)
        } else {
            self.num_levels_identical.partial_cmp(&other.num_levels_identical)
        }
    }
}

/*
// Do this if I want to make it work with KMeans++. Otherwise it is superflous.
impl Into<f64> for KernelPyramidDistance {
    fn into(self) -> f64 {
        todo!()
    }
}
 */

struct KernelPyramidImage {
    original: Image,
    indexed_kernel_images: Vec<Image>
}

impl KernelPyramidImage {
    pub fn top(&self) -> &Image {
        self.nth(0)
    }

    pub fn nth(&self, n: usize) -> &Image {
        &self.indexed_kernel_images[self.num_levels() - 1 - n]
    }

    pub fn num_levels(&self) -> usize {
        self.indexed_kernel_images.len()
    }

    pub fn distance(img1: &KernelPyramidImage, img2: &KernelPyramidImage) -> KernelPyramidDistance {
        assert_eq!(img1.original.side(), img2.original.side());
        assert_eq!(img1.num_levels(), img2.num_levels());
        let mut num_levels_identical = 0;
        while num_levels_identical < img1.num_levels() {
            let current_level_distance = hamming_distance(img1.nth(num_levels_identical), img2.nth(num_levels_identical));
            if current_level_distance == 0 {
                num_levels_identical += 1;
            } else {
                return KernelPyramidDistance { num_levels_identical, distance_level_n: current_level_distance }
            }
        }
        KernelPyramidDistance { num_levels_identical, distance_level_n: euclidean_distance(&img1.original, &img2.original) }
    }
}



fn kernel_stack_all(labeled_images: &Vec<(u8,Image)>, num_kernels: u8, num_levels: usize) -> Vec<(u8, KernelPyramidImage)> {
    let images_only = labeled_images.iter()
        .map(|(_,img)| img.clone())
        .collect();
    let kernels = extract_kernels_from(&images_only, num_kernels as usize,
                                       KERNEL_SIZE);

    let mut pyramid_images = labeled_images.iter()
        .map(|(label, img)| (*label, KernelPyramidImage {original: img.clone(),
            indexed_kernel_images: vec![indexed_kernel_image(&img, &kernels, &euclidean_distance)]}))
        .collect();

    for _ in 0..num_levels {
        add_pyramid_level(&mut pyramid_images, num_kernels);
    }

    pyramid_images
}

fn add_pyramid_level(pyramid_images: &mut Vec<(u8, KernelPyramidImage)>, num_kernels: u8) {
    let kernels = extract_indexed_kernels(pyramid_images, num_kernels);
    for (_, pyramid) in pyramid_images.iter_mut() {
        pyramid.indexed_kernel_images.push(indexed_kernel_image(&pyramid.top(), &kernels, &hamming_distance));
    }
}

fn hamming_distance(img1: &Image, img2: &Image) -> u32 {
    assert_eq!(img1.side(), img2.side());
    let mut distance = 0;
    for (x, y) in img1.x_y_iter() {
        if img1.get(x, y) != img2.get(x, y) {
            distance += 1;
        }
    }
    distance
}

fn index_mean(images: &Vec<&Image>) -> Image {
    assert!(!images.is_empty());
    assert!(images.iter().all(|img| img.len() == images[0].len()));

    let mut result = Image::new();
    for (x, y) in images[0].x_y_iter() {
        result.add(mode(&mut images.iter().map(|img| img.get(x, y))));
    }
    result
}

fn extract_indexed_kernels(pyramid_images: &Vec<(u8, KernelPyramidImage)>, num_kernels: u8) -> Vec<Image> {
    let mut candidates = Vec::new();
    for img in pyramid_images.iter().map(|(_, img)| img.top()) {
        add_kernels_from_to(img, &mut candidates, KERNEL_SIZE);
    }
    kmeans::Kmeans::new(num_kernels as usize, &candidates, hamming_distance, image_mean).move_means()
}

fn indexed_kernel_image<V: Copy + PartialEq + PartialOrd + Ord + Into<f64>, D: Fn(&Image,&Image) -> V>
(img: &Image, kernels: &Vec<Image>, distance: &D) -> Image {
    let mut result = Image::new();
    for (x, y) in img.x_y_step_iter(STRIDE) {
        result.add(classify_pixel(img, x, y, kernels, distance) as u8);
    }
    result
}

fn classify_pixel<V: Copy + PartialEq + PartialOrd + Ord, D: Fn(&Image,&Image) -> V>
(img: &Image, x: usize, y: usize, kernels: &Vec<Image>, distance: &D) -> usize {
    let (_, best_index) = kernels.iter()
        .enumerate()
        .map(|(i, kernel)| (distance(&img.subimage(x, y, KERNEL_SIZE), kernel), i))
        .min()
        .unwrap();
    best_index
}