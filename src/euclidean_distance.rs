//! The `euclidean_distance()` function takes two `Image` objects and returns their distance as a
//! `u32`, which is readily castable to an `f64` in the `kmeans` library.
//!
//! ```
//! use distance_research::euclidean_distance::euclidean_distance;
//! use distance_research::mnist_data::Image;
//!
//! let img1 = Image::from_vec(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
//! let img2 = Image::from_vec(&vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
//! assert_eq!(2 * (64 + 36 + 16 + 4), euclidean_distance(&img1, &img2));
//! ```

use crate::mnist_data::{Grid, Image};

// Using u32 so that it can be readily cast into an f64 in kmeans.
pub fn euclidean_distance(img1: &Image, img2: &Image) -> u32 {
    assert_eq!(img1.side(), img2.side());
    assert_eq!(img1.len(), img2.len());
    img1.x_y_iter()
        .map(|(x, y)| (img1.get(x, y) as i32 - img2.get(x, y) as i32).pow(2) as u32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist_data::Image;

    #[test]
    fn test_values() {
        let img1 = Image::from_vec(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let img2 = Image::from_vec(&vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(2 * (64 + 36 + 16 + 4), euclidean_distance(&img1, &img2));
    }
}