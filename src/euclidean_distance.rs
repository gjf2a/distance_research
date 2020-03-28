use crate::mnist_data::{Image, Grid};
use decorum::R64;

pub fn euclidean_distance(img1: &Image, img2: &Image) -> R64 {
    assert_eq!(img1.side(), img2.side());
    assert_eq!(img1.len(), img2.len());
    R64::from_inner(img1.x_y_iter()
        .map(|(x, y)| (img1.get(x, y) as f64 - img2.get(x, y) as f64).powf(2.0))
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_values() {
        let img1 = Image::from_vec(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let img2 = Image::from_vec(&vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(2.0 * (64.0 + 36.0 + 16.0 + 4.0), euclidean_distance(&img1, &img2).into_inner());
    }
}