extern crate decorum;
use crate::training_harness::Classifier;
use crate::hash_histogram::HashHistogram;

pub struct Knn<I, M, D: Fn(&I,&I) -> M> {
    k: usize,
    images: Vec<(u8,I)>,
    distance: D,
}

impl<I, M, D: Fn(&I,&I) -> M> Knn<I, M, D> {
    pub fn new(k: usize, distance: D) -> Knn<I, M, D> {
        Knn {k, images: Vec::new(), distance}
    }

    pub fn add_example(&mut self, img: (u8, I)) {
        self.images.push(img);
    }
}

impl<I: Clone, M: Copy + Eq + Ord, D: Fn(&I,&I) -> M> Classifier<I> for Knn<I, M, D> {
    fn train(&mut self, training_images: &Vec<(u8,I)>) {
        for img in training_images {
            // TODO: Bug report: self.add_example(img.clone()); // Flagged as type error by IDE, but compiles fine.
            self.add_example((img.0, img.1.clone()));
        }
    }

    fn classify(&self, example: &I) -> u8 {
        let mut distances: Vec<(M, u8)> = self.images.iter()
            .map(|img| ((self.distance)(example, &img.1), img.0))
            .collect();
        distances.sort();

        let mut labels = HashHistogram::new();
        for item in distances.iter().take(self.k) {
            labels.bump(item.1);
        }
        /*for i in 0..self.k {
            labels.bump(distances[i].1);
        }*/
        labels.mode()
    }
}