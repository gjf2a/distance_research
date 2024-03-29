use std::io;
use supervised_learning::Classifier;
use distance_research::mnist_data::{Image, load_data_set};
use std::env;
use std::collections::{HashSet, BTreeMap, HashMap};
use distance_research::brief::Descriptor;
use distance_research::convolutional::{kernelize_all, kernelized_distance};
use distance_research::kernel_patch::{kernelize_single_image, best_match_distance};
use distance_research::patch::patchify;
use distance_research::timing::print_time_milliseconds;
use distance_research::kernel_points::{find_keypoints, closest_for_all};
use distance_research::sobel::edge_image;
use distance_research::convolution_pyramid::{kernel_stack_all, KernelPyramidImage, get_kernels_from};

const SHRINK_SEQUENCE: [usize; 5] = [50, 20, 10, 5, 2];

const SHRINK_FACTOR: usize = 50;
const K: usize = 7;
const PATCH_SIZE: usize = 3;
const NUM_NEIGHBORS: usize = 8;
const CLASSIC_BRIEF_PAIRS: usize = distance_research::mnist_data::IMAGE_DIMENSION * distance_research::mnist_data::IMAGE_DIMENSION * NUM_NEIGHBORS;
const EQUIDISTANT_OFFSET: usize = distance_research::mnist_data::IMAGE_DIMENSION / 3;

const HELP: &str = "help";
const SHRINK: &str = "shrink";
const PERMUTE: &str = "permute";
const SEQ: &str = "sequence";

const BASELINE: &str = "baseline";
const BRIEF: &str = "brief";
const UNIFORM_BRIEF: &str = "uniform_brief";
const CONVOLUTIONAL_1: &str = "convolutional1";
const CONVOLUTIONAL_PYRAMID: &str = "convolutional_pyramid";
const PATCH: &str = "patch";
const UNIFORM_NEIGHBORS: &str = "uniform_neighbors";
const GAUSSIAN_NEIGHBORS: &str = "gaussian_neighbors";
const GAUSSIAN_7: &str = "gaussian_7";
const EQUIDISTANT_BRIEF: &str = "equidistant";
const EQUIDISTANT_3_3_BRIEF: &str = "equidistant_3_3";
const COMPARE_KERNELS: &str = "compare_kernels";
const COMPARE_KEYPOINTS: &str = "compare_keypoints";
const SOBEL_DIST: &str = "edge_distance";

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
    println!("\t{}: Convolutional pyramid", CONVOLUTIONAL_PYRAMID);
    println!("\t{}: Uniform Classical BRIEF descriptors", UNIFORM_BRIEF);
    println!("\t{}: Gaussian Classical BRIEF descriptors", BRIEF);
    println!("\t{}: 3x3 Neighbor BRIEF descriptors", PATCH);
    println!("\t{}: Uniform neighbor BRIEF", UNIFORM_NEIGHBORS);
    println!("\t{}: Gaussian neighbor BRIEF (stdev 1/3 side)", GAUSSIAN_NEIGHBORS);
    println!("\t{}: Gaussian neighbor BRIEF (stdev 1/7 side)", GAUSSIAN_7);
    println!("These variants are subsequent to the FLAIRS-2020 paper:");
    println!("\t{}: Equidistant BRIEF, where each pair consists of a pixel and another at a fixed x,y offset", EQUIDISTANT_BRIEF);
    println!("\t{}: Equidistant 3x3 kernel BRIEF, comparing 3x3 neighborhoods around the pixel pairs", EQUIDISTANT_3_3_BRIEF);
    println!("\t{}: Find 8 3x3 kernels for each image; add distance from each kernel to its best match", COMPARE_KERNELS);
    println!("\t{}: Find 8 3x3 kernels for each image; find 16 (x,y) points that best mach any of them; add distance from each point to its best match", COMPARE_KEYPOINTS);
    println!("\t{}: Euclidean distance between Sobel edge images", SOBEL_DIST);
}

fn train_and_test(args: &HashSet<String>) -> io::Result<()> {
    let mut training_images = load_data_set("train")?;
    let mut testing_images = load_data_set("t10k")?;

    if args.contains(SEQ) {
        for shrink in SHRINK_SEQUENCE.iter() {
            println!("Shrinking by {}", shrink);
            run_experiments(args, distance_research::mnist_data::discard(&training_images, *shrink),
                            distance_research::mnist_data::discard(&testing_images, *shrink))?;
        }

    } else {
        if args.contains(SHRINK) {
            println!("Shrinking by {}", SHRINK_FACTOR);
            training_images = distance_research::mnist_data::discard(&training_images, SHRINK_FACTOR);
            testing_images = distance_research::mnist_data::discard(&testing_images, SHRINK_FACTOR);
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

    data.add_descriptor(BRIEF, distance_research::brief::Descriptor::classic_gaussian_brief(CLASSIC_BRIEF_PAIRS, distance_research::mnist_data::IMAGE_DIMENSION, distance_research::mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(UNIFORM_BRIEF, distance_research::brief::Descriptor::classic_uniform_brief(CLASSIC_BRIEF_PAIRS, distance_research::mnist_data::IMAGE_DIMENSION, distance_research::mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(UNIFORM_NEIGHBORS, distance_research::brief::Descriptor::uniform_neighbor(NUM_NEIGHBORS, distance_research::mnist_data::IMAGE_DIMENSION, distance_research::mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(GAUSSIAN_NEIGHBORS, distance_research::brief::Descriptor::gaussian_neighbor(NUM_NEIGHBORS, distance_research::mnist_data::IMAGE_DIMENSION / 3, distance_research::mnist_data::IMAGE_DIMENSION, distance_research::mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(GAUSSIAN_7, distance_research::brief::Descriptor::gaussian_neighbor(NUM_NEIGHBORS, distance_research::mnist_data::IMAGE_DIMENSION / 7, distance_research::mnist_data::IMAGE_DIMENSION, distance_research::mnist_data::IMAGE_DIMENSION));
    data.add_descriptor(EQUIDISTANT_BRIEF, distance_research::brief::Descriptor::equidistant(distance_research::mnist_data::IMAGE_DIMENSION, distance_research::mnist_data::IMAGE_DIMENSION, EQUIDISTANT_OFFSET, EQUIDISTANT_OFFSET));

    data.run_all_tests_with(&args);

    if args.contains(PERMUTE) {
        println!("Permuting images");
        let permutation = distance_research::permutation::read_permutation("image_permutation_file")?;
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
    pub fn build_and_test_model<I: Clone, M: Copy + PartialEq + PartialOrd, C: Fn(&Image) -> I, D: Fn(&I,&I) -> M>
    (&mut self, label: &str, conversion: C, distance: D) {
        self.build_and_test_converting_all(label, |v| convert_all(v, &conversion), distance);
    }

    pub fn build_and_test_converting_all<I: Clone, M: Copy + PartialEq + PartialOrd, C: Fn(&Vec<(u8,Image)>) -> Vec<(u8,I)>, D: Fn(&I,&I) -> M>
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
        match self.descriptors.get(name) {
            Some(d) => d.clone(),
            None => {
                panic!("Descriptor {} not created", name)
            }
        }
    }

    pub fn add_descriptor(&mut self, name: &str, d: Descriptor) {
        self.descriptors.insert(name.to_string(), d);
    }

    pub fn run_all_tests_with(&mut self, args: &HashSet<String>) {
        if args.contains(BASELINE) {
            self.build_and_test_model(BASELINE, |v| v.clone(), distance_research::euclidean_distance::euclidean_distance);
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
        if args.contains(EQUIDISTANT_BRIEF) {
            self.build_and_test_descriptor(EQUIDISTANT_BRIEF);
        }
        if args.contains(EQUIDISTANT_3_3_BRIEF) {
            let descriptor = self.get_descriptor(EQUIDISTANT_BRIEF);
            self.build_and_test_model(EQUIDISTANT_3_3_BRIEF, |img| descriptor.apply_kernel(img, 3), bits::distance);
        }
        if args.contains(PATCH) {
            self.build_and_test_patch(PATCH, PATCH_SIZE);
        }
        if args.contains(CONVOLUTIONAL_1) {
            self.build_and_test_converting_all(CONVOLUTIONAL_1, |images| kernelize_all(images, 1), kernelized_distance);
        }
        if args.contains(CONVOLUTIONAL_PYRAMID) {
            let kernels = get_kernels_from(&self.training, 8);
            self.build_and_test_converting_all(CONVOLUTIONAL_PYRAMID, |images| kernel_stack_all(images, &kernels, 2), KernelPyramidImage::distance);
        }
        if args.contains(SOBEL_DIST) {
            self.build_and_test_converting_all(SOBEL_DIST, |images| images.iter().map(|(label, img)| (*label, edge_image(img))).collect(), distance_research::euclidean_distance::euclidean_distance);
        }
        if args.contains(COMPARE_KERNELS) {
            self.build_and_test_converting_all(COMPARE_KERNELS, |images| images.iter().map(|(label, img)| (*label, kernelize_single_image(img, 8, 3))).collect(), best_match_distance);
        }
        if args.contains(COMPARE_KEYPOINTS) {
            self.build_and_test_converting_all(COMPARE_KEYPOINTS, |images| images.iter().map(|(label, img)| (*label, find_keypoints(img, 8, 3, 64))).collect(), closest_for_all);
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