use crate::mnist_data::{Image, ImageIterator, Grid};
use crate::bits::BitArray;

pub fn patchify(img: &Image, patch_size: usize) -> BitArray {
    let mut patch = BitArray::new();
    for (x, y) in img.x_y_iter() {
        for (i, j) in ImageIterator::centered(x as isize, y as isize, patch_size as isize, patch_size as isize, 1) {
            patch.add(img.get(x, y) > img.option_get(i, j).unwrap_or(0));
        }
    }
    patch
}