use crate::mnist_data::{Image,Grid};

fn edge_pixel(img: &Image, x: usize, y: usize) -> u8 {
    compress(x_total(img, x, y) + y_total(img, x, y))
}

fn compress(edge_value: u16) -> u8 {
    if edge_value > u8::MAX as u16 {
        u8::MAX
    } else {
        edge_value as u8
    }
}

fn x_total(img: &Image, x: usize, y: usize) -> u16 {
    let mut total = 0;
    total -= get(img, x as isize - 1, y as isize - 1) as i16;
    total -= 2 * get(img, x as isize - 1, y as isize) as i16;
    total -= get(img, x as isize - 1, y as isize + 1) as i16;
    total += get(img, x as isize + 1, y as isize - 1) as i16;
    total += 2 * get(img, x as isize + 1, y as isize) as i16;
    total += get(img, x as isize + 1, y as isize + 1) as i16;
    total.abs() as u16
}

fn y_total(img: &Image, x: usize, y: usize) -> u16 {
    let mut total = 0;
    total -= get(img, x as isize - 1, y as isize - 1) as i16;
    total -= 2 * get(img, x as isize, y as isize - 1) as i16;
    total -= get(img, x as isize + 1, y as isize - 1) as i16;
    total += get(img, x as isize - 1, y as isize + 1) as i16;
    total += 2 * get(img, x as isize, y as isize + 1) as i16;
    total += get(img, x as isize + 1, y as isize + 1) as i16;
    total.abs() as u16
}

fn get(img: &Image, x: isize, y: isize) -> u8 {
    img.option_get(x, y).unwrap_or(0)
}

pub fn edge_image(img: &Image) -> Image {
    img.filter(edge_pixel)
}