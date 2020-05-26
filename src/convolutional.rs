#![allow(dead_code)]
use crate::mnist_data::{Image, Grid};
use kmeans;
use crate::euclidean_distance::euclidean_distance;
use itertools::{Itertools};
use std::rc::Rc;

const NUM_KERNELS: usize = 8;
const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;

pub fn kernelize_all<D: Clone + Default, T: Grid<D, V> + Clone + PartialEq,
                     V: Copy + PartialEq + PartialOrd + Into<f64>, F:Fn(&T,&T) -> V, M: Fn(&Vec<T>) -> T>
                     (labeled_images: &Vec<(u8, T)>, levels: usize, distance: F, mean: M) -> Vec<(u8, Vec<T>)>{
    let mut images: Vec<Rc<T>>= Vec::new();
    for (_, img) in labeled_images{
        images.push(Rc::new(img.clone()));
    }
   let kernels = extract_kernels_from(&images, NUM_KERNELS, KERNEL_SIZE, &distance, mean);

    let mut kernelized: Vec<(u8,Vec<T>)> = Vec::new();
    for (label, img) in labeled_images{
        kernelized.push((*label, vec![img.clone()]))
    }
    for _ in 0..levels {
        kernelized = kernelized.iter().map(|(label, images)| (*label, project_all_through(images, &kernels, &distance))).collect_vec();
    }
    kernelized
}

pub fn kernelized_distance(k1: &Vec<Image>, k2: &Vec<Image>) -> f64 {
    assert_eq!(k1.len(), k2.len());
    (0..k1.len()).map(|i| euclidean_distance(&k1[i], &k2[i])).sum()

}

pub fn extract_kernels_from
    <D: Clone + Default, T: Grid<D, V> + Clone + PartialEq , V: Copy + PartialEq + PartialOrd + Into<f64>, F:Fn(&T,&T) -> V, M: Fn(&Vec<T>) -> T>
    (images: &Vec<Rc<T>>, num_kernels: usize, kernel_size: usize, distance: &F, mean: M) -> Vec<T> {
    let mut candidates: Vec<T> = Vec::new();
    for img in images.iter() {
        add_kernels_from_to(img, &mut candidates, kernel_size);
    }
    kmeans::Kmeans::new(num_kernels, &candidates.to_owned(), distance, mean).move_means()
}

pub fn project_all_through<D: Clone + Default, T: Grid<D, V> + Clone + PartialEq,
                           V: Copy + PartialEq + PartialOrd + Into<f64>, F:Fn(&T,&T) -> V>
                           (images: &Vec<T>, kernels: &Vec<T>, distance: &F) -> Vec<T> {
    let mut result = Vec::new();
    for img in images.iter() {
        result.append(&mut project_image_through(Rc::new(img.to_owned()), kernels, &distance));
    }
    result
}

pub fn project_image_through<D: Clone + Default, T: Grid<D, V> + Clone + PartialEq,
                             V: Copy + PartialEq + PartialOrd, F:Fn(&T,&T) -> V>
                             (img: Rc<T>, kernels: &Vec<T>, distance: &F) -> Vec<T> {
    kernels.iter().map(|kernel| apply_kernel_to(&img, &Rc::new(kernel.to_owned()), &distance)).collect_vec()
}

pub fn apply_kernel_to<D: Clone + Default, T: Grid<D, V> + Clone + PartialEq,
                       V: Copy + PartialEq + PartialOrd, F:Fn(&T,&T) -> V>
                       (img: &Rc<T>, kernel: &Rc<T>, distancefn: &F) -> T {
    let mut result: T  = img.default();
    for(x, y) in img.x_y_step_iter(STRIDE){
        let subimg = &img.subimage(x, y, KERNEL_SIZE);
        let distance: V  = distancefn(subimg, kernel);
        result.add(img.pixelize(distance, KERNEL_SIZE));
    }
    result
}

//R64 for the distance
//distance as an integer value

//pixelize -> takes the input of the output of the distance function
//if the distance is below the value then return false
//picking a threshold value as an input parameter, threshold testing
//median distance

//make generic to something else than u8
//take a distance and turn it into a bit
//mean distance for all pixels and then threshold for 0 and 1 below and above
//threshold needs to be set
//avoid f64 keep everything as the base
//pub fn pixelize(distance: R64) -> boolean
//threshold has a higher a higher order function |threshold| -> distance value
//was it the same or different
pub fn pixelize(distance: f64) -> u8 {
    //                       ((255^2 = 65,025) * (3^2 as f64))^(.5)
    //                       (biggest gap between two pixels is 255) * (number of pixels in the kernel)
    let max_distance = ((std::u8::MAX as f64).powf(2.0) * (KERNEL_SIZE.pow(2) as f64)).powf(0.5);
    let distance_to_pixel_scale = (std::u8::MAX as f64) / max_distance;
    //(units squared) * (the scaling factor) = to fit into the 255
    (distance.powf(0.5) * distance_to_pixel_scale) as u8
}

fn add_kernels_from_to<D: Clone + Default, T: Grid<D, U> + Clone + PartialEq, U>(img: &Rc<T>, raw_filters: &mut Vec<T>, kernel_size: usize) {
    img.x_y_iter().
        for_each(|(x, y)| raw_filters.push(img.subimage(x, y, kernel_size)));
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;
    use crate::mnist_data::image_mean;
    use bits::BitArray;
    use std::fmt::Binary;

    #[test]
    fn test_kernels() {
        let img = Image::from_vec(&(1..10).collect());
        let filters = extract_kernels_from(&vec![Rc::new(img)], 4, 2, &euclidean_distance, image_mean);
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

    #[test]
    fn test_apply_kernel_to(){
        const SIZE: usize = 2;
        let img = Image::from_vec(&vec![2,4,6,8]);
        print_image(&img);
        println!("*********************");
        let img_clone = img.clone();
        // let filters = extract_kernels_from(&vec![img], 4, 2);
        // let mut count = 1;
        // for img in &filters {
        //     println!("Image {} -> ", count);
        //     print_image(&img);
        //     count += 1;
        // }
        // println!("*********************");
        println!("Apply First Kernel to Image");
        println!();
        let kernel = Image::from_vec(&vec![1, 2, 3, 4]);
        print_image(&kernel);
        let distance = euclidean_distance(&img_clone, &kernel);
        println!();
        println!("Distance between first kernel and image at 0,0: {}", distance);
        let conversion = pixelize(distance);
        println!();
        println!("u8: {}", conversion);
    }

    fn print_image(img: &Image) { &img.x_y_iter().
                                    for_each(|(x, y)|
                                        println!("x: {}, y: {}, val: {}", x, y, &img.get(x,y))); }

    #[test]
    fn binarized_image_test(){
        let mut b1 = BitArray::new();
        let mut b2 = BitArray::new();
        b1.add(true);
        b2.add(false);
        b1.add(true);
        b2.add(false);
        let diff: u32 = 2;
        for _ in 0..(BitArray::word_size() - 1) {
            b1.add(false);
            b2.add(false);
        }
        print_binimage(&b1);
    }
    fn print_binimage(img: &BitArray) { &img.x_y_iter().
        for_each(|(x, y)|
            println!("x: {}, y: {}, val: {}", x, y, &img.get(x,y))); }

}