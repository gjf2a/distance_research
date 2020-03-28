mod mnist_data;
mod training_harness;
mod knn;
mod hash_histogram;
mod euclidean_distance;
mod permutation;
mod brief;
mod kmeans;
mod patch;
mod convolutional;
mod bits;
mod timing;

use std::io;
use crate::training_harness::Classifier;
use crate::mnist_data::Image;
use std::env;
use std::collections::{HashSet, BTreeMap, HashMap};
use crate::brief::Descriptor;
use crate::convolutional::{kernelize_all, kernelized_distance};
use crate::patch::patchify;
use crate::timing::print_time_milliseconds;

const SHRINK_SEQUENCE: [usize; 5] = [50, 20, 10, 5, 2];

const BASE_PATH: &str = "/Users/ferrer/Desktop/mnist_data/";
const SHRINK_FACTOR: usize = 50;
const K: usize = 7;
const PATCH_SIZE: usize = 3;
const NUM_NEIGHBORS: usize = 8;
const CLASSIC_BRIEF_PAIRS: usize = mnist_data::IMAGE_DIMENSION * mnist_data::IMAGE_DIMENSION * NUM_NEIGHBORS;

const HELP: &str = "help";
const SHRINK: &str = "shrink";
const PERMUTE: &str = "permute";
const SEQ: &str = "sequence";

const BASELINE: &str = "baseline";
const BRIEF: &str = "brief";
const UNIFORM_BRIEF: &str = "uniform_brief";
const CONVOLUTIONAL_1: &str = "convolutional1";
const PATCH: &str = "patch";
const UNIFORM_NEIGHBORS: &str = "uniform_neighbors";
const GAUSSIAN_NEIGHBORS: &str = "gaussian_neighbors";
const GAUSSIAN_7: &str = "gaussian_7";

fn main() -> io::Result<()> {
    let args: HashSet<String> = env::args().collect();
    if args.contains(HELP) {
        help_message();
    } else {
        train_and_test(&args)?;
    }
    Ok(())
}

fn help_message() {
    println!("Usage: flairs33 [options]:");
    println!("\t{}: print this message", HELP);
    println!("\t{}: runs additional experiment that permutes image pixels", PERMUTE);
    println!("\t{}: Use only 1 out of {} training/testing images", SHRINK, SHRINK_FACTOR);
    println!("\t{}: Use 1/50, 1/20, 1/10, 1/5, and 1/2 training/testing images", SEQ);
    println!("\nAlgorithmic options:");
    println!("The eight variants of the paper are given in order of appearance in Tables 1 and 2.");
    println!("All variants describe a knn (k=7) distance function variation:");
    println!("\t{}: Euclidean", BASELINE);
    println!("\t{}: Convolutional Euclidean (1 level)", CONVOLUTIONAL_1);
    println!("\t{}: Uniform Classical BRIEF descriptors", UNIFORM_BRIEF);
    println!("\t{}: Gaussian Classical BRIEF descriptors", BRIEF);
    println!("\t{}: 3x3 Neighbor BRIEF descriptors", PATCH);
    println!("\t{}: Uniform neighbor BRIEF", UNIFORM_NEIGHBORS);
    println!("\t{}: Gaussian neighbor BRIEF (stdev 1/3 side)", GAUSSIAN_NEIGHBORS);
    println!("\t{}: Gaussian neighbor BRIEF (stdev 1/7 side)", GAUSSIAN_7);
}

fn train_and_test(args: &HashSet<String>) -> io::Result<()> {
    let mut training_images = load_data_set("train")?;
    let mut testing_images = load_data_set("t10k")?;

    if args.contains(SEQ) {
        for shrink in SHRINK_SEQUENCE.iter() {
            println!("Shrinking by {}", shrink);
            run_experiments(args, mnist_data::discard(&training_images, *shrink),
                            mnist_data::discard(&testing_images, *shrink))?;
        }

    } else {
        if args.contains(SHRINK) {
            println!("Shrinking by {}", SHRINK_FACTOR);
            training_images = mnist_data::discard(&training_images, SHRINK_FACTOR);
            testing_images = mnist_data::discard(&testing_images, SHRINK_FACTOR);
        }

        run_experiments(args, training_images, testing_images)?;
    }

    Ok(())
}

fn run_experiments(args: &HashSet<String>, training_images: Vec<(u8,Image)>, testing_images: Vec<(u8,Image)>) -> io::Result<()> {
    let mut data = ExperimentData {
        training: training_images,
        testing: testing_images,
        descriptors: Default::default(),
        errors: BTreeMap::new()
    };

    data.add_descriptor(BRIEF, brief::Descriptor::classic_gaussian_brief(CLASSIC_BRIEF_PAIRS, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(UNIFORM_BRIEF, brief::Descriptor::classic_uniform_brief(CLASSIC_BRIEF_PAIRS, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(UNIFORM_NEIGHBORS, brief::Descriptor::uniform_neighbor(NUM_NEIGHBORS, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(GAUSSIAN_NEIGHBORS, brief::Descriptor::gaussian_neighbor(NUM_NEIGHBORS, mnist_data::IMAGE_DIMENSION / 3, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(GAUSSIAN_7, brief::Descriptor::gaussian_neighbor(NUM_NEIGHBORS, mnist_data::IMAGE_DIMENSION / 7, mnist_data::IMAGE_DIMENSION, mnist_data::IMAGE_DIMENSION));

    data.run_all_tests_with(&args);

    if args.contains(PERMUTE) {
        println!("Permuting images");
        let permutation = permutation::read_permutation("image_permutation_file")?;
        let mut permuted_data = data.permuted(&permutation);
        permuted_data.run_all_tests_with(&args);
        println!("Permuted results");
        permuted_data.print_errors();
        println!();
    }

    println!("Original results");
    data.print_errors();
    Ok(())
}

fn load_data_set(file_prefix: &str) -> io::Result<Vec<(u8,Image)>> {
    let train_images = format!("{}{}-images-idx3-ubyte", BASE_PATH, file_prefix);
    let train_labels = format!("{}{}-labels-idx1-ubyte", BASE_PATH, file_prefix);

    let training_images = print_time_milliseconds(&format!("loading mnist {} images", file_prefix),
        || mnist_data::init_from_files(train_images.as_str(), train_labels.as_str()))?;

    println!("Number of {} images: {}", file_prefix, training_images.len());
    Ok(training_images)
}

fn permuted_data_set(permutation: &Vec<usize>, data: &Vec<(u8,Image)>) -> Vec<(u8,Image)> {
    data.iter()
        .map(|(label, img)| (*label, img.permuted(permutation)))
        .collect()
}

fn convert_all<I, C: Fn(&Image) -> I>(labeled_list: &Vec<(u8, Image)>, conversion: C) -> Vec<(u8, I)> {
    labeled_list.iter().map(|(label, img)| (*label, conversion(img))).collect()
}

#[derive(Clone)]
pub struct ExperimentData {
    training: Vec<(u8,Image)>,
    testing: Vec<(u8,Image)>,
    descriptors: HashMap<String,Descriptor>,
    errors: BTreeMap<String,f64>
}

impl ExperimentData {
    pub fn build_and_test_model<I: Clone, M: Copy + Eq + Ord, C: Fn(&Image) -> I, D: Fn(&I,&I) -> M>
    (&mut self, label: &str, conversion: C, distance: D) {
        self.build_and_test_converting_all(label, |v| convert_all(v, &conversion), distance);
    }

    pub fn build_and_test_converting_all<I: Clone, M: Copy + Eq + Ord, C: Fn(&Vec<(u8,Image)>) -> Vec<(u8,I)>, D: Fn(&I,&I) -> M>
    (&mut self, label: &str, conversion: C, distance: D) {
        let training_images = print_time_milliseconds(&format!("converting training images to {}", label),
                                                      || conversion(&self.training));

        let testing_images = print_time_milliseconds(&format!("converting testing images to {}", label),
                                                     || conversion(&self.testing));

        let mut model = knn::Knn::new(K, distance);
        print_time_milliseconds(&format!("training {} model (k={})", label, K),
                                || model.train(&training_images));
        let outcome = print_time_milliseconds("testing", || model.test(&testing_images));
        print!("{}", outcome);
        let error_percentage = outcome.error_rate() * 100.0;
        println!("Error rate: {}", error_percentage);
        self.errors.insert(label.to_string(), error_percentage);
    }

    pub fn get_descriptor(&self, name: &str) -> Descriptor {
        self.descriptors.get(name).unwrap().clone()
    }

    pub fn add_descriptor(&mut self, name: &str, d: Descriptor) {
        self.descriptors.insert(name.to_string(), d);
    }

    pub fn run_all_tests_with(&mut self, args: &HashSet<String>) {
        if args.contains(BASELINE) {
            self.build_and_test_model(BASELINE, |v| v.clone(), euclidean_distance::euclidean_distance);
        }
        if args.contains(BRIEF) {
            self.build_and_test_descriptor(BRIEF);
        }
        if args.contains(UNIFORM_BRIEF) {
            self.build_and_test_descriptor(UNIFORM_BRIEF);
        }
        if args.contains(UNIFORM_NEIGHBORS) {
            self.build_and_test_descriptor(UNIFORM_NEIGHBORS);
        }
        if args.contains(GAUSSIAN_NEIGHBORS) {
            self.build_and_test_descriptor(GAUSSIAN_NEIGHBORS);
        }
        if args.contains(GAUSSIAN_7) {
            self.build_and_test_descriptor(GAUSSIAN_7);
        }
        if args.contains(PATCH) {
            self.build_and_test_patch(PATCH, PATCH_SIZE);
        }
        if args.contains(CONVOLUTIONAL_1) {
            self.build_and_test_converting_all(CONVOLUTIONAL_1, |images| kernelize_all(images, 1), kernelized_distance);
        }
    }

    fn build_and_test_descriptor(&mut self, descriptor_name: &str) {
        let descriptor = self.get_descriptor(descriptor_name);
        self.build_and_test_model(descriptor_name, |img| descriptor.apply_to(img), bits::distance);
    }

    fn build_and_test_patch(&mut self, label: &str, patch_size: usize) {
        self.build_and_test_model(label, |img| patchify(img, patch_size), bits::distance);
    }

    pub fn permuted(&self, permutation: &Vec<usize>) -> ExperimentData {
        ExperimentData {
            training: permuted_data_set(permutation, &self.training),
            testing: permuted_data_set(permutation, &self.testing),
            descriptors: self.descriptors.clone(),
            errors: BTreeMap::new()
        }
    }

    pub fn print_errors(&self) {
        for (k,v) in self.errors.iter() {
            println!("{}: {}%", k, v);
        }
    }
}