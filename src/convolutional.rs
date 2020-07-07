#![allow(dead_code)]
use crate::mnist_data::{Image, Grid};
use kmeans;
use crate::euclidean_distance::euclidean_distance;
use itertools::{Itertools};
use std::rc::Rc;
use bits::BitArray;
use crate::hamming_distance::hamming_distance;
use soc::means::traits::{Means, PartialCmp};
extern crate image;



const NUM_KERNELS: usize = 8;
const KERNEL_SIZE: usize = 3;
const STRIDE: usize = 2;

pub trait Replicate = Clone + Default;
pub trait CData<D: Default,V> = Grid<D, V> + Means;

pub fn kernelize_all<D: Replicate, T: CData<D, V>, V: PartialCmp + Into<f64>, F:Fn(&T,&T) -> V, M: Fn(&Vec<T>) -> T>
                     (labeled_images: &Vec<(u8, T)>, levels: usize, distance: F, mean: M) -> Vec<(u8, Vec<T>)>{
    let mut images = labeled_images.iter().map(|(_,img)| img);
    let kernels = extract_kernels_from2(images, NUM_KERNELS, KERNEL_SIZE, &distance, mean);
    let mut kernelized: Vec<(u8,Vec<T>)> = Vec::new();
    for (label, img) in labeled_images{
        kernelized.push((*label, vec![img.clone()]))
    }
    for _ in 0..levels {
        kernelized = kernelized.iter().map(|(label, images)| (*label, project_all_through(images, &kernels, &distance))).collect();
    }
    kernelized
}

fn print_all_kernels<D: Replicate, T: CData<D, V>,
V: PartialCmp + Into<f64>>(kernels: &Vec<T>, name: String) {
    for (i, img) in kernels.iter().enumerate(){
        let name = format!("{}_{}.png", name, i);
        img.save(name).unwrap();
    }
}

pub fn extract_kernels_from2<'a, D: Clone + Default, T: 'a + Grid<D, V> + Means, V: PartialCmp + Into<f64>,
                            F:Fn(&T,&T) -> V, M: Fn(&Vec<T>) -> T>
            (images: impl Iterator<Item = &'a T>, num_kernels: usize, kernel_size: usize, distance: &F, mean: M) -> Vec<T>{
    let mut candidates: Vec<T> = Vec::new();
    for img in images {
        add_kernels_from_to(img, &mut candidates, kernel_size);
    }
    soc::SOCluster::new_trained(num_kernels, &candidates, distance, mean).move_clusters()
    //kmeans::Kmeans::new(num_kernels, &candidates, distance, mean).move_means()

}

pub fn kernelized_distance(k1: &Vec<Image>, k2: &Vec<Image>) -> f64 {
    assert_eq!(k1.len(), k2.len());
    (0..k1.len()).map(|i| euclidean_distance(&k1[i], &k2[i])).sum()

}

pub fn kernelized_dist_bitarray(b1: &Vec<BitArray>, b2: &Vec<BitArray>) -> u32{
    assert_eq!(b1.len(), b2.len());
    (0..b1.len()).map(|i| hamming_distance(&b1[i], &b2[i])).sum()
}

pub fn extract_kernels_from
    <D: Clone + Default, T: Grid<D, V> + Means, V: PartialCmp + Into<f64>, F:Fn(&T,&T) -> V, M: Fn(&Vec<T>) -> T>
    (images: &Vec<T>, num_kernels: usize, kernel_size: usize, distance: &F, mean: M) -> Vec<T> {
    let length = images.len();
    println!("length: {}", length);
    let mut candidates: Vec<T> = Vec::new();
    for img in images{
        add_kernels_from_to(img, &mut candidates, kernel_size);
    }
    println!("ADDED KERNELS");
    soc::SOCluster::new_trained(num_kernels, &candidates, distance, mean).move_clusters()
    //kmeans::Kmeans::new(num_kernels, &candidates, distance, mean).move_means()
}

pub fn project_all_through<D: Clone + Default, T: Grid<D, V> + Means,
                           V: PartialCmp + Into<f64>, F:Fn(&T,&T) -> V>
                           (images: &Vec<T>, kernels: &Vec<T>, distance: &F) -> Vec<T> {
    let mut result = Vec::new();
    for img in images.iter() {
        result.append(&mut project_image_through(img, kernels, &distance));
    }

    result
}

pub fn project_image_through<D: Clone + Default, T: Grid<D, V> + Means,
                             V: PartialCmp, F:Fn(&T,&T) -> V>
                             (img: &T, kernels: &Vec<T>, distance: &F) -> Vec<T> {
    kernels.iter().map(|kernel| apply_kernel_to(img, kernel, &distance)).collect()
}

pub fn apply_kernel_to<D: Clone + Default, T: Grid<D, V> + Means,
                       V: PartialCmp, F:Fn(&T,&T) -> V>
                       (img: &T, kernel: &T, distancefn: &F) -> T {
    let mut result: T  = img.default();
    for(x, y) in img.x_y_step_iter(STRIDE){
        let subimg = &img.subimage(x, y, KERNEL_SIZE);
        let distance: V  = distancefn(subimg, kernel);
        result.add(img.pixelize(distance, KERNEL_SIZE));
    }
    result
}

pub fn pixelize(distance: f64) -> u8 {
    //                       ((255^2 = 65,025) * (3^2 as f64))^(.5)
    //                       (biggest gap between two pixels is 255) * (number of pixels in the kernel)
    let max_distance = ((std::u8::MAX as f64).powf(2.0) * (KERNEL_SIZE.pow(2) as f64)).powf(0.5);
    let distance_to_pixel_scale = (std::u8::MAX as f64) / max_distance;
    //(units squared) * (the scaling factor) = to fit into the 255
    (distance.powf(0.5) * distance_to_pixel_scale) as u8
}

fn add_kernels_from_to<D: Clone + Default, T: Grid<D, U> + Means, U: PartialCmp>(img: &T, raw_filters: &mut Vec<T>, kernel_size: usize) {
    img.x_y_iter().
        for_each(|(x, y)| raw_filters.push(img.subimage(x, y, kernel_size)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist_data::image_mean;
    use bits::BitArray;
    use crate::hamming_distance::hamming_distance;
    use crate::{mnist_data, K};
    use image::image_dimensions;
    use supervised_learning::Classifier;

    #[test]
    fn test_kernels() {
        let img = Image::from_vec(&(1..10).collect());
        let filters = extract_kernels_from(&vec![img], 4, 2, &euclidean_distance, image_mean);
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

    fn run_kernelize(labeled_images: &Vec<(u8, Image)>) -> Vec<(u8, Vec<Image>)>{
        println!("Images: {:?}", labeled_images);
        let images = labeled_images.iter().map(|(_,img)| img.clone()).collect();
        let kernels = extract_kernels_from(&images, NUM_KERNELS, KERNEL_SIZE, &euclidean_distance, image_mean);
        let mut kernelized = test_create_kernelized();
        let kernelized_out: Vec<(u8,Vec<Image>)> = kernelized.iter().map(|(label, images)| (*label, project_all_through(images, &kernels, &euclidean_distance))).collect();
        assert_eq!(kernelized.len(), kernelized_out.len());
        for (i, (label, imgs)) in kernelized.iter().enumerate() {
            let (k_label, k_imgs) = kernelized_out.get(i).unwrap();
            assert_eq!(NUM_KERNELS, k_imgs.len());
            assert_eq!(k_label, label);
            assert_eq!(1, imgs.len());
            //println!("kernelized_{}: {:?}", i, k_imgs);
        }
        kernelized_out
    }

    #[test]
    fn test_classify() {
        let images = gen_images();
        let kernels_1 = run_kernelize(&images);
        let mut model = knn::Knn::new(K, kernelized_distance);
        model.train(&kernels_1);
        for (label, imgs) in kernels_1{
            let guess = model.classify(&imgs);
            println!("label: {}, guess: {}", label, guess);
        }
    }

    fn test_create_kernelized() -> Vec<(u8, Vec<Image>)>{
        let labeled_images = gen_images();
        let mut kernelized: Vec<(u8,Vec<Image>)> = labeled_images.iter().map(|(label, img)| (*label, vec![img.clone()])).collect();
        assert_eq!(labeled_images.len(), kernelized.len());
        for (i, (label, img)) in labeled_images.iter().enumerate(){
            let (k_label, k_imgs) = kernelized.get(i).unwrap();
            assert_eq!(1, k_imgs.len());
            let k_img = k_imgs.get(0).unwrap();
            assert_eq!(img, k_img);
            assert_eq!(label, k_label);
        }
        kernelized
    }

    fn gen_images() -> Vec<(u8, Image)>{
        const IMAGE_NUM: usize = 30;
        let mut kernels: Vec<(u8, Image)> = Vec::new();
        for i in 0..IMAGE_NUM{
            let vec: Vec<usize> = (1..10).collect();
            let vec2: Vec<u8> = vec.iter().map(|val| (val + i) as u8).collect();
            let image = Image::from_vec(&vec2);
            let label = i as u8;
            kernels.push((label, image));
        }
        kernels
    }

    #[test]
    fn test_extract_kernels_from(){
        let labeled_images = gen_images();
        println!("Images: {:?}", labeled_images);
        let images = labeled_images.iter().map(|(_,img)| img.clone()).collect();
        let kernels = extract_kernels_from(&images, NUM_KERNELS, KERNEL_SIZE,&euclidean_distance, image_mean);
        assert_eq!(NUM_KERNELS, kernels.len());
        let sample = kernels.get(0).unwrap();
        assert_eq!(KERNEL_SIZE, sample.side());
        assert_eq!(KERNEL_SIZE * KERNEL_SIZE, sample.len());
        println!("Kernels: {:?}", kernels);
    }

    #[test]
    fn add_kernel_from_to(){
        let image = Image::from_vec(&(1..10).collect());
        let length = 9;
        assert_eq!(length, image.len());
        let mut candidates: Vec<Image> = Vec::new();
        add_kernels_from_to(&image, &mut candidates, KERNEL_SIZE);
        assert_eq!(length, candidates.len());
        let sample = candidates.get(0).unwrap();
        assert_eq!(KERNEL_SIZE, sample.side());
        assert_eq!(KERNEL_SIZE * KERNEL_SIZE, sample.len());
        for (i, img) in candidates.iter().enumerate(){
            println!("img_{}: {:?}", i, img);
        }
    }

    #[test]
    fn test_project_image_through(){
        const KERNELS: &str = "[Image { pixels: [0, 2, 4, 5], side_size: 2 }, Image { pixels: [2, 0, 4, 4], side_size: 2 }, Image { pixels: [2, 3, 4, 5], side_size: 2 }, Image { pixels: [3, 4, 5, 5], side_size: 2 }, Image { pixels: [4, 2, 5, 4], side_size: 2 }, Image { pixels: [4, 4, 0, 4], side_size: 2 }, Image { pixels: [5, 5, 2, 3], side_size: 2 }, Image { pixels: [5, 4, 4, 0], side_size: 2 }]";
        const NUM_KERNELS: usize = 8;
        const KERNEL_SIZE: usize = 3;
        let image = Image::from_vec(&(1..10).collect());
        let image2 = Image::from_vec(&(10..19).collect());
        let images: Vec<Image> = vec![image.clone(), image2.clone()];
        let kernels = extract_kernels_from(&images, NUM_KERNELS, KERNEL_SIZE, &euclidean_distance, image_mean);
        let mut kernelized = vec![("l1".as_ptr() as u8, vec![image.clone()]), ("l2".as_ptr() as u8, vec![image2.clone()])];
        let k1 = kernelized.iter().map( |(label, images)| (*label, project_all_through(images, &kernels, &euclidean_distance))).collect_vec();
        let k2: Vec<(u8, Vec<Image>)> = kernelized.iter().map( |(label, images)| (*label, project_all_through(images, &kernels, &euclidean_distance))).collect();
        for (i, (u, vec)) in k1.iter().enumerate(){
            let (u2, image) = k2.get(i).unwrap();
            assert_eq!(u, u2);
            assert_eq!(vec, image);
        }
    }



    // #[test]
    // fn test_add_kernels_from_to(){
    //     const SIZE: usize = 3;
    //     let img = Image::from_vec(&vec![1,1,1,1,0,1,1,1,1]);
    //     let bin_img = build_binimage(&vec![true, true, true, true, false, true, true , true, true]);
    //     let mut img_cand: Vec<Image> = Vec::new();
    //     add_kernels_from_to(&img, &mut img_cand, 2);
    //     let mut bin_img_cand: Vec<BitArray> = Vec::new();
    //     add_kernels_from_to(&bin_img, &mut bin_img_cand, 2);
    //     assert_eq!(&img_cand.len(), &bin_img_cand.len());
    //     println!("IMAGES: {}", &img_cand.len());
    //     for img in img_cand.iter(){
    //         print_image(img);
    //         println!();
    //     }
    //     println!("BINIMAGES: {}", &bin_img_cand.len());
    //     for bin_img in bin_img_cand.iter(){
    //         print_binimage(bin_img);
    //         println!();
    //     }
    //
    // }

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

    fn build_binimage(image: &Vec<bool>) -> BitArray{
        let mut b1 = BitArray::new();
        for i in 0..image.len(){
            mnist_data::Grid::add(&mut b1, image[i])
        }
        b1
    }

    #[test]
    fn binarized_image_test(){
        let mut b1 = BitArray::new();
        let mut b2 = BitArray::new();
        // b1: 1 1 0 0 1 1 0 1 -> 5 ones
        // b2: 1 1 1 1 1 1 1 1 -> 8 ones
        // distance: 0 0 1 1 0 0 1 0 = 3
        // pixelize: 1 1 0 0 1 1 0 1 -> 5 < 3 -> false
        let vex1 = vec![true, true, false, false, true, true, false, true];
        let vex2 = vec![true, true, true, true, true, true, true, true];
        assert_eq!(true, vex1.len() == vex2.len());
        for i in 0..vex1.len(){
            mnist_data::Grid::add(&mut b1, vex1[i].clone());
            mnist_data::Grid::add(&mut b2, vex2[i].clone());
        }
        print_binimage(&b1);
        println!();
        print_binimage(&b2);
        println!();
        let distance = hamming_distance(&b1, &b2);
        println!();
        assert_eq!(3, distance);
        let conversion = b1.pixelize(distance, BitArray::word_size());
        assert_eq!(false, conversion);

    }
    fn print_binimage(img: &BitArray) { &img.x_y_iter().
        for_each(|(x, y)|
            println!("x: {}, y: {}, val: {}", x, y, &img.get(x,y))); }


}