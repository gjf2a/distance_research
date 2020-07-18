use std::fs;
use std::io;
use std::io::Read;
use std::ops::{AddAssign, Add};
use bits::*;
use soc::means::traits::Means;
use image::ImageError;

pub const IMAGE_DIMENSION: usize = 28;
pub const IMAGE_BYTES: usize = IMAGE_DIMENSION * IMAGE_DIMENSION;

#[derive(Clone, Debug, Default)]
pub struct Image {
    pixels: Vec<u8>,
    side_size: usize,
}

impl Means for Image{
    fn calc_mean(&self, other: &Image) -> Image {
        image_mean(&vec![self.clone(), other.clone()])
    }

    fn calc_means(vals: &Vec<Self>) -> Self {
        image_mean(vals)
    }

    fn calc_weighted(&self, w1: f64, w2: f64, other: &Self) -> Self {
        assert_eq!(self.pixels.len(), other.pixels.len());
        //assert that tge sum is <0 an >255
        let mut sums: Vec<f64> = Vec::new();
        for p in 0..self.pixels.len() {
                sums.push( (w1 * self.pixels[p] as f64) + (w2 * other.pixels[p] as f64));
        }
        let mut result = Image::new();
        for sum in sums {
            assert!(sum >= 0.0);
            assert!(sum <= 255.0);
            result.add(sum as u8);
        }
        result
    }
}

pub trait Grid<T: Default, DIS> {

    fn add(&mut self, pixel: T);
    fn get(&self, x: usize, y: usize) -> T;
    fn side(&self) -> usize;
    fn len(&self) -> usize;


    fn in_bounds(&self, x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && x < self.side() as isize && y < self.side() as isize
    }

    fn option_get(&self, x: isize, y: isize) -> Option<T> {
        if self.in_bounds(x, y) { Some(self.get(x as usize, y as usize)) } else { None }
    }

    fn x_y_iter(&self) -> ImageIterator<usize> {
        ImageIterator::new(0, 0, self.side(), self.side(), 1)
    }

    fn x_y_step_iter(&self, step_size: usize) -> ImageIterator<usize> {
        ImageIterator::new(0, 0, self.side(), self.side(), step_size)
    }

    fn subimage(&self, x_center: usize, y_center: usize, side: usize) -> Self;

    fn default(&self) -> Self;

    fn pixelize(&self, distance: DIS, kernel_size: usize) -> T;

    fn save(&self, mes: String) -> Result<(), ImageError>;
}

impl Grid<bool, u32> for BitArray{

    fn add(&mut self, pixel: bool) {
        self.add(pixel)
    }

    fn get(&self, x: usize, y: usize) -> bool {
        assert!(self.in_bounds(x as isize, y as isize));
        self.is_set((y * self.side() + x) as u64)
    }

    fn side(&self) -> usize {
        //Assuming a square
        (self.len() as f64).sqrt() as usize
    }

    fn len(&self) -> usize {
        self.len() as usize
    }

    fn subimage(&self, x_center: usize, y_center: usize, side: usize) -> BitArray {
        let mut result = BitArray::new();
        ImageIterator::centered(x_center as isize, y_center as isize, side as isize, side as isize, 1)
            .for_each(|(x, y)| result.add(self.option_get(x, y).unwrap_or(false)));
        result
    }

    fn default(&self) -> BitArray {
        BitArray::new()
    }

    fn pixelize(&self, distance: u32, _kernel_size: usize) -> bool {
        let ones = self.count_bits_on();
        ones < distance
    }

    fn save(&self, mes: String) -> Result<(), ImageError>{
        unimplemented!()
    }
}

impl Grid<u8, f64> for Image {

    fn add(&mut self, pixel: u8) {
        self.pixels.push(pixel);
        if self.pixels.len() > self.side_size.pow(2) {
            self.side_size += 1;
        }
    }

    fn get(&self, x: usize, y: usize) -> u8 {
        assert!(self.in_bounds(x as isize, y as isize));
        self.pixels[y * self.side_size + x]
    }

    fn side(&self) -> usize {
        self.side_size
    }

    fn len(&self) -> usize {
        self.pixels.len()
    }

    fn subimage(&self, x_center: usize, y_center: usize, side: usize) -> Image {
        let mut result = Image::new();
        ImageIterator::centered(x_center as isize, y_center as isize, side as isize, side as isize, 1)
            .for_each(|(x, y)| result.add(self.option_get(x, y).unwrap_or(0)));
        assert_eq!(side, result.side());
        result
    }

    fn default (&self) -> Image{
        Image::new()
    }

    fn pixelize(&self, distance: f64, kernel_size: usize) -> u8 {
        let max_distance = ((std::u8::MAX as f64).powf(2.0) * (kernel_size.pow(2) as f64)).powf(0.5);
        let distance_to_pixel_scale = (std::u8::MAX as f64) / max_distance;
        //(units squared) * (the scaling factor) = to fit into the 255
        (distance.powf(0.5) * distance_to_pixel_scale) as u8
    }

    fn save(&self, name: String) -> Result<(), ImageError>{
        let w = self.side_size as u32;
        let name = format!("centroid_{}.png", name);
        image::save_buffer(name, self.pixels.as_ref(), w, w,image::ColorType::L8)
    }
}

impl Image {
    pub fn new() -> Self {
        Default::default()
    }

    #[cfg(test)]
    pub fn from_vec(v: &Vec<u8>) -> Image {
        let mut result = Image::new();
        v.iter().for_each(|p| result.add(*p));
        result
    }

    pub fn permuted(&self, permutation: &[usize]) -> Image {
        assert_eq!(self.pixels.len(), permutation.len());
        let mut result = Image::new();
        for index in permutation {
            result.add(self.pixels[*index]);
        }
        result
    }

    pub fn shrunken(&self, shrink: usize) -> Image {
        let mut result = Image::new();
        let target_side = self.side() / shrink;
        ImageIterator::new(0, 0, target_side, target_side, 1)
            .for_each(|(x, y)| result.add(self.subimage_mean(x, y, shrink)));
        result
    }

    pub fn subimage(&self, x_center: usize, y_center: usize, side: usize) -> Image {
        let mut result = Image::new();
        ImageIterator::centered(x_center as isize, y_center as isize, side as isize, side as isize, 1)
            .for_each(|(x, y)| result.add(self.option_get(x, y).unwrap_or(0)));
        assert_eq!(side, result.side());
        result
    }

    fn subimage_mean(&self, x: usize, y: usize, side: usize) -> u8 {
        let mut sum: u16 = 0;
        for i in x..x + side {
            for j in y..y + side {
                sum += self.get(i, j) as u16;
            }
        }
        (sum / side.pow(2) as u16) as u8
    }

    #[cfg(test)]
    pub fn pixel_mean(&self) -> u8 {
        let mut sum: u16 = 0;
        self.x_y_iter().for_each(|(x, y)| sum += self.get(x, y) as u16);
        (sum / self.pixels.len() as u16) as u8
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.side_size == other.side_size && self.pixels.len() == other.pixels.len() && (0..self.pixels.len()).all(|i| self.pixels[i] == other.pixels[i])
    }
}

impl Eq for Image {}

pub fn bitarray_max(images: &Vec<BitArray>) -> BitArray {
    assert!(!images.is_empty());
    assert!(images.iter().all(|img| img.len() == images[0].len()));
    let mut sums: Vec<usize> = (0..images[0].len()).map(|_| 0).collect();
    for image in images.iter(){
        for (x,y) in image.x_y_iter() {
            let index = (y * image.side()) + x;
            sums[index] += bits_to_num(image.get(x, y));
        }
    }
    let mut result = BitArray::new();
    let images_length = images.len();
    for ones in sums {
        let zeros = images_length - ones;
        //println!("ones: {}, zeros: {}", ones, zeros);
        result.add(ones >= zeros);
    }
    result
}

fn bits_to_num(b: bool) -> usize{
    if b{
        1
    }else {
        0
    }
}

pub fn image_mean(images: &Vec<Image>) -> Image {
    assert!(!images.is_empty());
    assert!(images.iter().all(|img| img.pixels.len() == images[0].pixels.len()));
    let mut sums: Vec<usize> = (0..images[0].pixels.len()).map(|_| 0).collect();
    for image in images.iter() {
        for p in 0..image.pixels.len() {
            sums[p] += image.pixels[p] as usize;
        }
    }

    let mut result = Image::new();
    for sum in sums {
        result.add((sum / images.len()) as u8);
    }
    result
}

pub fn image_mean_borrowed(images: &Vec<&Image>) -> Image {
    assert!(!images.is_empty());
    assert!(images.iter().all(|img| img.pixels.len() == images[0].pixels.len()));
    let mut sums: Vec<usize> = (0..images[0].pixels.len()).map(|_| 0).collect();
    for image in images.iter() {
        for p in 0..image.pixels.len() {
            sums[p] += image.pixels[p] as usize;
        }
    }

    let mut result = Image::new();
    for sum in sums {
        result.add((sum / images.len()) as u8);
    }
    result
}

pub struct ImageIterator<N> {
    width: N,
    height: N,
    x: N,
    y: N,
    x_start: N,
    y_start: N,
    stride: N
}

impl<N: Copy> ImageIterator<N> {
    pub fn new(x: N, y: N, width: N, height: N, stride: N) -> ImageIterator<N> {
        ImageIterator {x, y, width, height, x_start: x, y_start: y, stride}
    }
}

impl ImageIterator<isize> {
    pub fn centered(x: isize, y: isize, width: isize, height: isize, stride: isize) -> ImageIterator<isize> {
        ImageIterator::new(x - width / 2, y - height / 2, width, height, stride)
    }
}

impl<N: Copy + AddAssign + Add<Output=N> + Eq + Ord> Iterator for ImageIterator<N> {
    type Item = (N,N);

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.y_start + self.height {
            None
        } else {
            let result = (self.x, self.y);
            self.x += self.stride;
            if self.x >= self.x_start + self.width {
                self.x = self.x_start;
                self.y += self.stride;
            }
            Some(result)
        }
    }
}

pub fn init_from_files(image_file_name: &str, label_file_name: &str) -> io::Result<Vec<(u8,Image)>> {
    let bytes: Vec<u8> = read_label_file(label_file_name)?;
    read_image_file(image_file_name, &bytes)
}

pub fn discard(items: &[(u8,Image)], shrink: usize) -> Vec<(u8,Image)> {
    let mut result: Vec<(u8,Image)> = Vec::new();
    for i in 0..items.len() {
        if i % shrink == 0 {
            result.push(items[i].clone())
        }
    }
    result
}

fn read_label_file(label_file_name: &str) -> io::Result<Vec<u8>> {
    let fin = fs::File::open(label_file_name)?;
    let mut bytes: Vec<u8> = Vec::new();
    let mut header_bytes_left = 8;
    for b in fin.bytes() {
        if header_bytes_left > 0 {
            header_bytes_left -= 1;
        } else {
            let b = b?;
            bytes.push(b);
        }
    }
    Ok(bytes)
}

fn read_image_file(image_file_name: &str, labels: &[u8]) -> io::Result<Vec<(u8,Image)>> {
    let fin = fs::File::open(image_file_name)?;
    let mut images: Vec<(u8,Image)> = Vec::new();
    let mut image: Image = Image::new();
    let mut header_bytes_left = 16;
    for b in fin.bytes() {
        if header_bytes_left > 0 {
            header_bytes_left -= 1;
        } else {
            image.add(b?);
            if image.len() == IMAGE_BYTES {
                images.push((labels[images.len()], image));
                image = Image::new();
            }
        }
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::print_time_milliseconds;
    use crate::load_data_set;
    extern crate image;

    const BASE_PATH: &str = "/home/david86/Y3C/DistanceMetrics/distance_research/mnist_data2/";

    #[test]
    fn print_img(){
        let train = "train";
        let test = "t10k";
        let train_images = load_data_set2(train).unwrap();
        let train_labels = load_data_set2(test).unwrap();
        let (u, img) = train_images.get(0).unwrap();
        let w = img.side() as u32;
        let h = img.side() as u32;
        image::save_buffer("test.png", img.pixels.as_ref(), w, h,image::ColorType::L8).unwrap();



    }

    fn load_data_set2(file_prefix: &str) -> io::Result<Vec<(u8,Image)>> {
        let train_images = format!("{}{}-images-idx3-ubyte", BASE_PATH, file_prefix);
        let train_labels = format!("{}{}-labels-idx1-ubyte", BASE_PATH, file_prefix);


        let training_images = print_time_milliseconds(&format!("loading mnist {} images", file_prefix),
                                                      || init_from_files(train_images.as_str(), train_labels.as_str()))?;


        println!("Number of {} images: {}", file_prefix, training_images.len());
        Ok(training_images)
    }

    #[test]
    fn test_img() {
        let mut img = Image::new();
        assert_eq!(0, img.side());
        img.add(10);
        assert_eq!(1, img.side());
        assert_eq!(10, img.get(0, 0));
        img.add(20);
        assert_eq!(2, img.side());
        assert_eq!(20, img.get(1, 0));
        img.add(30);
        assert_eq!(2, img.side());
        assert_eq!(30, img.get(0, 1));
        img.add(40);
        assert_eq!(2, img.side());
        assert_eq!(40, img.get(1, 1));
        img.add(50);
        assert_eq!(3, img.side());
        assert_eq!(30, img.get(2, 0));
        assert_eq!(40, img.get(0, 1));
        assert_eq!(50, img.get(1, 1));
    }

    #[test]
    fn test_x_y_iterator_1() {
        let iterated: Vec<(isize,isize)> = ImageIterator::new(0, 0, 2, 3, 1).collect();
        let reference: Vec<(isize,isize)> = vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)];
        assert_eq!(iterated.len(), reference.len());
        for i in 0..reference.len() {
            assert_eq!(reference[i], iterated[i]);
        }
    }

    #[test]
    fn test_x_y_iterator_2() {
        let iterated: Vec<(isize,isize)> = ImageIterator::new(-1, -1, 2, 2, 1).collect();
        let reference: Vec<(isize,isize)> = vec![(-1, -1), (0, -1), (-1, 0), (0, 0)];
        assert_eq!(iterated.len(), reference.len());
        for i in 0..reference.len() {
            assert_eq!(reference[i], iterated[i]);
        }
    }

    #[test]
    fn test_subimage() {
        let img = Image::from_vec(&(1..16).collect());
        let sub = img.subimage(1, 1, 3);
        let ref_sub = Image::from_vec(&vec![1, 2, 3, 5, 6, 7, 9, 10, 11]);
        assert_eq!(ref_sub, sub);
    }

    // 1 1 0 0 -> 2 -> true
    // 1 1 1 0 -> 3 -> true
    // 1 0 0 0 -> 1 -> false
    // 1 1 1 1 -> 4 -> true    // 0 0 0 0 -> 0 -> false
    #[test]
    fn test_image_mean_bitarray(){
        let b1 = build_binimage(&vec![true, false, true, false]);
        let b2 = build_binimage(&vec![true, false, true, false]);
        let b3 = build_binimage(&vec![true, false, false, false]);
        let b4 = build_binimage(&vec![false, true, false, false]);
        //print_binimage(&b1);
        //println!();
        //print_binimage(&b2);
        let bins = vec![b1.clone(), b2.clone(), b3.clone(), b4.clone()];
        let means = bitarray_max(&bins);
        //println!();
        //print_binimage(&result);
        let result = build_binimage(&vec![true, false, true, false]);
        assert_eq!(result, means);
    }


    fn build_binimage(image: &Vec<bool>) -> BitArray{
        let mut b1 = BitArray::new();
        for i in 0..image.len(){
            b1.add(image[i]);

        }
        b1
    }

    fn print_binimage(img: &BitArray) {
        &img.x_y_iter().for_each(|(x, y)| println!("x: {}, y: {}, val: {}", x, y, &img.get(x,y)));
    }

}