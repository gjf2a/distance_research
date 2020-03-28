use crate::mnist_data::{Image, image_mean, Grid};
use crate::kmeans;
use decorum::R64;
use crate::euclidean_distance::euclidean_distance;

const NUM_KERNELS: usize = 8;
const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;

pub fn kernelize_all(labeled_images: &Vec<(u8,Image)>, levels: usize) -> Vec<(u8,Vec<Image>)> {
    let kernels = extract_kernels_from(&(labeled_images.iter().map(|(_,img)| img.clone()).collect()), NUM_KERNELS, KERNEL_SIZE);
    let mut kernelized: Vec<(u8,Vec<Image>)> = labeled_images.iter().map(|(label, img)| (*label, vec![img.clone()])).collect();
    for _ in 0..levels {
        kernelized = kernelized.iter().map(|(label, images)| (*label, project_all_through(images, &kernels))).collect();
    }
    kernelized
}

pub fn kernelized_distance(k1: &Vec<Image>, k2: &Vec<Image>) -> R64 {
    assert_eq!(k1.len(), k2.len());
    (0..k1.len()).map(|i| euclidean_distance(&k1[i], &k2[i])).sum()
}

pub fn extract_kernels_from(images: &Vec<Image>, num_kernels: usize, kernel_size: usize) -> Vec<Image> {
    let mut candidates = Vec::new();
    for img in images.iter() {
        add_kernels_from_to(img, &mut candidates, kernel_size);
    }
    kmeans::Kmeans::new(num_kernels, &candidates, euclidean_distance, image_mean).move_means()
}

pub fn project_all_through(images: &Vec<Image>, kernels: &Vec<Image>) -> Vec<Image> {
    let mut result = Vec::new();
    for img in images.iter() {
        result.append(&mut project_image_through(img, kernels));
    }
    result
}

pub fn project_image_through(img: &Image, kernels: &Vec<Image>) -> Vec<Image> {
    kernels.iter().map(|kernel| apply_kernel_to(img, kernel)).collect()
}

pub fn apply_kernel_to(img: &Image, kernel: &Image) -> Image {
    assert_eq!(kernel.side(), KERNEL_SIZE);
    let mut result = Image::new();
    for (x, y) in img.x_y_step_iter(STRIDE) {
        result.add(pixelize(euclidean_distance(&img.subimage(x, y, KERNEL_SIZE), kernel)));
    }
    result
}

pub fn pixelize(distance: R64) -> u8 {
    let max_distance = ((std::u8::MAX as f64).powf(2.0) * (KERNEL_SIZE.pow(2) as f64)).powf(0.5);
    let distance_to_pixel_scale = (std::u8::MAX as f64) / max_distance;
    (distance.into_inner().powf(0.5) * distance_to_pixel_scale) as u8
}

fn add_kernels_from_to(img: &Image, raw_filters: &mut Vec<Image>, kernel_size: usize) {
    img.x_y_iter().
        for_each(|(x, y)| raw_filters.push(img.subimage(x, y, kernel_size)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernels() {
        let img = Image::from_vec(&(1..10).collect());
        let filters = extract_kernels_from(&vec![img], 4, 2);
        let filter_means: Vec<u8> = filters.iter().map(|f| f.pixel_mean()).collect();

        let target_means_1: Vec<u8> = vec![3, 0, 6, 1];
        let target_means_2: Vec<u8> = vec![3, 0, 6, 7];
        assert!(test_filter_means(&target_means_1, &filter_means) ||
                test_filter_means(&target_means_2, &filter_means));
    }

    fn test_filter_means(target_means: &Vec<u8>, filter_means: &Vec<u8>) -> bool {
        for mean in filter_means.iter() {
            if !target_means.contains(mean) && !target_means.contains(&(mean - 1)) && !target_means.contains(&(mean + 1)) {
                return false;
            }
        }
        true
    }
}