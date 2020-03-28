use crate::hash_histogram::HashHistogram;
use std::{fmt, io};
use std::fmt::Formatter;
use std::collections::HashSet;
use std::io::Write;

pub struct ConfusionMatrix {
    label_2_right: HashHistogram<u8>,
    label_2_wrong: HashHistogram<u8>,
}

impl ConfusionMatrix {
    pub fn new() -> ConfusionMatrix {
        ConfusionMatrix {
            label_2_right: HashHistogram::new(),
            label_2_wrong: HashHistogram::new(),
        }
    }

    pub fn record(&mut self, img_label: u8, classification: u8) {
        if classification == img_label {
            self.label_2_right.bump(img_label);
        } else {
            self.label_2_wrong.bump(img_label);
        }
    }

    pub fn all_labels(&self) -> HashSet<u8> {
        self.label_2_wrong.all_labels()
            .union(&self.label_2_right.all_labels())
            .copied()
            .collect()
    }

    pub fn error_rate(&self) -> f64 {
        let total_right = self.label_2_right.total_count() as f64;
        let total_wrong = self.label_2_wrong.total_count() as f64;
        total_wrong / (total_right + total_wrong)
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut ordered_labels: Vec<u8> = self.all_labels().iter().copied().collect();
        ordered_labels.sort_unstable();
        for label in ordered_labels {
            writeln!(f, "{}: {} correct, {} incorrect", label, self.label_2_right.get(label), self.label_2_wrong.get(label))?;
        }
        Ok(())
    }
}

pub trait Classifier<I> {
    fn train(&mut self, training_images: &Vec<(u8,I)>);

    fn classify(&self, example: &I) -> u8;

    fn test(&self, testing_images: &[(u8,I)]) -> ConfusionMatrix {
        let mut result = ConfusionMatrix::new();
        let mut count = 0;
        let twentieth = testing_images.len() / 20;
        for test_img in testing_images {
            result.record(test_img.0, self.classify(&test_img.1));
            count += 1;
            if count % twentieth == 0 {
                print!("{}%; ", count * 5 / twentieth);
                io::stdout().flush().expect("Could not flush stdout");
            }
        }
        println!();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let mut matrix = ConfusionMatrix::new();

        let one_ok = 6;
        let one_er = 4;
        let two_ok = 7;
        let two_er = 3;

        for _ in 0..one_ok {
            matrix.record(1, 1);
        }

        for _ in 0..one_er {
            matrix.record(1, 2);
        }

        for _ in 0..two_ok {
            matrix.record(2, 2);
        }

        for _ in 0..two_er {
            matrix.record(2, 1);
        }

        assert_eq!(format!("1: {} correct, {} incorrect\n2: {} correct, {} incorrect\n", one_ok, one_er, two_ok, two_er), matrix.to_string());
    }
}