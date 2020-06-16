use bits::{distance, BitArray};

pub fn hamming_distance(img1: &BitArray, img2: &BitArray) -> u32 {
    distance(img1, img2)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_values(){
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
        assert_eq!(diff, hamming_distance(&b1, &b2))
    }
}

