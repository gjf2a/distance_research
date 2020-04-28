#![allow(dead_code)]
use crate::mnist_data::{Image, image_mean, Grid};
use kmeans;
use decorum::R64;
use crate::euclidean_distance::euclidean_distance;
use itertools::{cloned, Itertools};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;


const NUM_KERNELS: usize = 8;
const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;

///TODO: generalize this kernalize all to take in label images as &Vec(u8, BitArray) or &Vec(u8, BinarizedImage)
/// generic of Grid<T>

pub fn kernelize_all<D: Clone + Default, T: Grid<D> + Clone + Eq>(labeled_images: &Vec<(u8, T)>, levels: usize) -> Vec<(u8, Vec<T>)>{
    let mut images: Vec<Rc<T>>= Vec::new();
    for (_, img) in labeled_images{
        images.push(Rc::new(img.clone()));
    }
   let kernels = extract_kernels_from(&images, NUM_KERNELS, KERNEL_SIZE);

    let mut kernelized: Vec<(u8,Vec<T>)> = Vec::new();
    for (label, img) in labeled_images{
        kernelized.push((*label, vec![img.clone()]))
    }
    //if kernels returns 8 convoluted images then kernalized contains for each labeled image a vec of the convolutions of each kernal

    //kernalized would be a Vec < Label, 8 convoluted images of the current labeled image> and we return this
    // for _ in 0..levels {
    //     kernelized = kernelized.iter().map(|(label, images)| (*label, project_all_through(images, &kernels))).collect();
    // }
    // kernelized

    let mut img = Image::new();
    img.add(0);
    let temp: Vec<(u8, Vec<T>)> = vec![(0, Vec::new())];
    temp
}

// pub fn kernelize_all(labeled_images: &Vec<(u8,Image)>, levels: usize) -> Vec<(u8,Vec<Image>)> {
//     let kernels = extract_kernels_from(&(labeled_images.iter().map(|(_,img)| img.clone()).collect()), NUM_KERNELS, KERNEL_SIZE);
//     let mut kernelized: Vec<(u8,Vec<Image>)> = labeled_images.iter().map(|(label, img)| (*label, vec![img.clone()])).collect();
//     //if kernels returns 8 convoluted images then kernalized contains for each labeled image a vec of the convolutions of each kernal
//     //kernalized would be a Vec < Label, 8 convoluted images of the current labeled image> and we return this
//     for _ in 0..levels {
//         kernelized = kernelized.iter().map(|(label, images)| (*label, project_all_through(images, &kernels))).collect();
//     }
//     kernelized
// }

pub fn kernelized_distance(k1: &Vec<Image>, k2: &Vec<Image>) -> R64 {
    assert_eq!(k1.len(), k2.len());
    (0..k1.len()).map(|i| euclidean_distance(&k1[i], &k2[i])).sum()
}

pub fn extract_kernels_from<D: Clone + Default, T: Grid<D> + Clone + Eq>
                            (images: &Vec<Rc<T>>, num_kernels: usize, kernel_size: usize) -> Vec<T> {
    let mut candidates: Vec<T> = Vec::new();
    for img in images.iter() {
        add_kernels_from_to(img, &mut candidates, kernel_size);
    }
    
    kmeans::Kmeans::new(num_kernels, &candidates.to_owned(), euclidean_distance, image_mean).move_means()
}

pub fn project_all_through(images: &Vec<Image>, kernels: &Vec<Image>) -> Vec<Image> {
    let mut result = Vec::new();
    //takes the vec of images and then iterates over all images
    //for each image we apply project image through
    //project image through takes the image and kernels
    //for each kernel we apply all kernels in the Vec to the current image
    //the kernels are created from all images as subimages
    for img in images.iter() {
        result.append(&mut project_image_through(img, kernels));
    }
    result
}

pub fn project_image_through(img: &Image, kernels: &Vec<Image>) -> Vec<Image> {
    //this returns a list of convolved images for each kernel of the one current image
    kernels.iter().map(|kernel| apply_kernel_to(img, kernel)).collect()
}

pub fn apply_kernel_to(img: &Image, kernel: &Image) -> Image {
    //this is the actual convolution of image and kernel
    assert_eq!(kernel.side(), KERNEL_SIZE);
    let mut result = Image::new();
    for (x, y) in img.x_y_step_iter(STRIDE) {
        //we find the distance between the current subimage with the current kernel
        //which is the same for all subimages looped through
        //pixelize just converts the distance to a u8 using propotions of u8 and f64 sizes somehow????
        result.add(pixelize(euclidean_distance(&img.subimage(x, y, KERNEL_SIZE), kernel)));
        //TODO: generalize this distance function to work with any Fn(Image, Image) -> R64
        // euclidean_distance becomes hamming distance
    }
    result
}

//make generic to something else than u8
//take a distance and turn it into a bit
//mean distance for all pixels and then threshold for 0 and 1 below and above
//threshold needs to be set
//avoid f64 keep everything as the base
//pub fn pixelize(distance: R64) -> boolean
//threshold has a higher a higher order function |threshold| -> distance value
//was it the same or different
pub fn pixelize(distance: R64) -> u8 {
    //                       ((255^2 = 65,025) * (3^2 as f64))^(.5)
    //                       (biggest gap between two pixels is 255) * (number of pixels in the kernel)
    let max_distance = ((std::u8::MAX as f64).powf(2.0) * (KERNEL_SIZE.pow(2) as f64)).powf(0.5);
    let distance_to_pixel_scale = (std::u8::MAX as f64) / max_distance;
    //(units squared) * (the scaling factor) = to fit into the 255
    (distance.into_inner().powf(0.5) * distance_to_pixel_scale) as u8
}

fn add_kernels_from_to<D: Clone + Default, T: Grid<D> + Clone + Eq>(img: &Rc<T>, raw_filters: &mut Vec<T>, kernel_size: usize) {
    img.x_y_iter().
        for_each(|(x, y)| raw_filters.push(img.subimage(x, y, kernel_size)));
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Borrow;

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
        println!("Distance between first kernel and image at 0,0: {}", distance.into_inner());
        let conversion = pixelize(distance);
        println!();
        println!("u8: {}", conversion);





    }

    fn print_image(img: &Image) { &img.x_y_iter().
                                    for_each(|(x, y)|
                                        println!("x: {}, y: {}, val: {}", x, y, &img.get(x,y))); }

}