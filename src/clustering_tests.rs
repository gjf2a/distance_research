use std::f64::NAN;
use std::{thread, time};
use crate::mnist_data::Image;
use rand::thread_rng;
use rand::distributions::{Uniform, Distribution};
use soc::means::traits::Means;

pub fn classify<T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V>(target: &T, means: &Vec<T>, distance: &D) -> (V, usize) {
    let distances: Vec<(V,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]).into(), i))
        .collect();
    let y = distances.iter()
        .fold(None, |m:Option<&(V, usize)>, d| m.map_or(Some(d), |m|
            Some(if m.0 < d.0 {m} else {d}))).unwrap();
    (y.0, y.1)
}

fn build_random_clusters(k: usize, data: &Vec<Image>) -> (Vec<Image>, Vec<usize>){
    let (centroids, images) = data.split_at(k);
    let mut result = centroids.to_vec();
    let mut counts: Vec<usize> = (0..k).map(|_| 0).collect();
    let mut rng = thread_rng();
    let uniform = Uniform::new(0, k);
    for image in images{
        let removal = uniform.sample(&mut rng);
        counts[removal] += 1;
        result.insert(removal, image.clone());
    }
    (result, counts)
}

fn random_clustering(k: usize, data: &Vec<Image>) -> Vec<Image>{
    let (centroids, images) = data.split_at(k);
    let mut result = centroids.to_vec();
    let mut counts: Vec<usize> = (0..k).map(|_| 0).collect();
    let mut rng = thread_rng();
    let uniform = Uniform::new(0, k);
    for (i, image) in images.iter().enumerate(){
        let removal = uniform.sample(&mut rng);
        counts[removal] += 1;
        result[removal] = result[removal].calc_mean(image);
        println!("{}", i);
    }
    result
}

pub mod tests{
    use super::*;
    use soc::*;
    use kmeans::*;
    use std::{io};
    use crate::mnist_data::{Image, init_from_files, image_mean, Grid, image_mean_borrowed};
    use crate::timing::print_time_milliseconds;
    use crate::euclidean_distance::euclidean_distance;
    use hash_histogram::HashHistogram;
    use std::collections::BTreeSet;
    use std::cmp::{max, min};
    use itertools::Itertools;
    use std::sync::{Arc, Mutex};

    const BASE_PATH: &str = "/Users/david/Desktop/RustProjects/distance_research/distance_research/mnist_data2/";
    const NUM_CENTROIDS: usize = 8;
    const NUM_TRAINING: usize = 60000;
    const NUM_TESTING: usize = 10000;

    //TODO: Make all images through a clustering done at the beginning

    //mean medium and stdev
    //just go with kmeans initalizer -> classifier
    //add new methods for weighting

    fn load_data_set2(file_prefix: &str) -> io::Result<Vec<(u8,Image)>> {
        let train_images = format!("{}{}-images-idx3-ubyte", BASE_PATH, file_prefix);
        let train_labels = format!("{}{}-labels-idx1-ubyte", BASE_PATH, file_prefix);

        let training_images = print_time_milliseconds(&format!("loading mnist {} images", file_prefix),
                                                      || init_from_files(train_images.as_str(), train_labels.as_str()))?;

        println!("Number of {} images: {}", file_prefix, training_images.len());
        Ok(training_images)
    }

    fn get_images(name: &str) -> Vec<Image> {
        let train_images = load_data_set2(name).unwrap();
        train_images.iter().map(|(_,img)| img.clone()).collect()
    }

    #[test]
    fn test_random_clustering(){
        println!("testing before");
        let train = "train";
        let train_images = load_data_set2(train).unwrap();
        let images: Vec<Image>= train_images.iter().map(|(_,img)| img.clone()).collect();
        println!("testing after");
        let means = random_clustering(NUM_CENTROIDS, &images);
        classify_label(&train_images, &means);
    }


    #[test]
    fn run_random_cluster_picker(){
        let images = get_images("t10k");
        assert_eq!(images.len(), 10000);
        let (left, right) = images.split_at(100);
        let (left2, right2) = right.split_at(1000);
        let (clusters3, counts3) = build_random_clusters(NUM_CENTROIDS, &left.to_vec());
        let (clusters2, counts2) = build_random_clusters(NUM_CENTROIDS, &left2.to_vec());
        let (clusters, counts) = build_random_clusters(NUM_CENTROIDS, &images);
        println!("Unifom Counts 10000: {:?}", counts);
        println!("Unifom Counts 1000: {:?}", counts2);
        println!("Unifom Counts 100: {:?}", counts3);
    }

    #[test]
    fn test_kmeans(){
        let images = get_images("train");
        let kmeans = print_time_milliseconds("Kmeans++ clustering",
                                          || Kmeans::new(NUM_CENTROIDS, &images, euclidean_distance, image_mean_borrowed));
        println!();
        let kmeans_means: Vec<Image> = kmeans.move_means();
        println!();
        clustering_results_means_dist(&images, &kmeans_means, "kmeans");
    }


    #[test]
    fn test_soc_weighted(){
        let images = get_images("train");
        let soc = print_time_milliseconds("SOC clustering",
                                            || SOCluster::new_trained(NUM_CENTROIDS, &images, euclidean_distance));
        println!();
        let counts = soc.copy_counts();
        println!("soc counts: {:?}", counts);
        let soc_means: Vec<Image> = soc.move_clusters();
        println!();
        clustering_results_means_dist(&images, &soc_means, "soc");
    }


    #[test]
    fn test_soc_unweighted(){
        let images = get_images("train");
        let soc = print_time_milliseconds("SOC unweighted clustering",
                                          || SOCluster::new_trained_unweighted(NUM_CENTROIDS, &images, euclidean_distance));
        println!();
        let counts = soc.copy_counts();
        println!("soc unweighted counts: {:?}", counts);
        let soc_means: Vec<Image> = soc.move_clusters();
        println!();
        clustering_results_means_dist(&images, &soc_means, "soc");
    }

    #[test]
    fn test_uniform_random(){
        println!("FIRST!");
        let images = get_images("train");
        println!("{}", images.len());
        // let uniform = build_random_clusters(NUM_CENTROIDS, &images);
        // println!("Counts: ");
        // get_clustering_counts("Uniform Random", &images, &uniform.0);
        // println!();
        // println!("Means: ");
        // clustering_results_means_dist(&images, &uniform.0, "Uniform Random");

    }

    fn clustering_results_means_dist(images: &Vec<Image>, means: &Vec<Image>, name: &str){

        assert_eq!(NUM_CENTROIDS, means.len());

        let mut clustered: Vec<Vec<f64>> = (0..NUM_CENTROIDS).map(|_| Vec::new()).collect();;

        for img in images {
            let closest: (f64, usize) = classify(img, means, &euclidean_distance);
            clustered[closest.1].push(closest.0);
        }

        let mut dist_mean: Vec<f64> = Vec::new();

        for (i, vec) in clustered.iter().enumerate(){
            let sum: f64 = vec.iter().sum();
            let mean: f64 = sum / (vec.len() as f64);
            dist_mean.push(mean);
            println!("{} cluster {} mean distance: {}", name, i, mean);
        }

        println!();

        let average_mean: f64 = dist_mean.iter().sum::<f64>() / (dist_mean.len() as f64);
        println!("{} all clusters average mean distance: {}", name, average_mean);

        let mut dist_var: Vec<f64> = Vec::new();

        println!();

        for (i, vec) in clustered.iter().enumerate(){
            let v_sum = vec.iter().fold(0.0, |sum, val| sum + ((val - dist_mean[i]).powf(2.0)));
            let variance = v_sum / (clustered.len() as f64);
            dist_var.push(variance);
            println!("{} cluster {} variance: {}", name, i, variance);
        }

        println!();

        let average_var: f64 = dist_var.iter().sum::<f64>() / (dist_var.len() as f64);
        println!("{} all clusters average variance distance: {}", name, average_var);
    }

    #[test]
    fn test_all_clusters() {
        //8 clusters 10000 images
        let mut handles = vec![];
        let load = get_images("t10k");
        println!("finished loading");
        let images = Arc::new(load);

        let b =Arc::clone(&images);
        let handle2 = thread::spawn( move|| test_cluster_counts_soc_weighted(&b));
        handles.push(handle2);

        let b =Arc::clone(&images);
        let handle3 = thread::spawn( move|| test_cluster_counts_soc_unweighted(&b));
        handles.push(handle3);

        let b =Arc::clone(&images);
        let handle1 = thread::spawn( move|| test_cluster_counts_kmeans(&b));
        handles.push(handle1);

        for handle in handles{
            handle.join().unwrap();
        }


    }

    fn test_cluster_counts_kmeans(images: &Vec<Image>){
        let kmeans = print_time_milliseconds("Kmeans++ clustering",
                                             || Kmeans::new(NUM_CENTROIDS, images, euclidean_distance, image_mean_borrowed));
        println!();
        let means = kmeans.move_means();

        get_clustering_counts("Kmeans++ clustering", &images, &means);
    }

    fn test_cluster_counts_soc_weighted(images: &Vec<Image>){
        let soc = print_time_milliseconds("SOC weighted clustering",
                                            || SOCluster::new_trained(NUM_CENTROIDS, images, euclidean_distance));
        let counts = soc.copy_counts();
        let means = soc.move_clusters();
        println!("Direct Counts: {:?}", counts);

        println!();
        get_clustering_counts("SOC weighted", &images, &means);
    }

    fn test_cluster_counts_soc_unweighted(images: &Vec<Image>){
        let soc = print_time_milliseconds("SOC unweighted clustering",
                                          || SOCluster::new_trained_unweighted(NUM_CENTROIDS, images, euclidean_distance));
        let counts = soc.copy_counts();
        let means = soc.move_clusters();
        println!("Direct Counts: {:?}", counts);

        println!();
        get_clustering_counts("SOC unweighted", &images, &means);
    }

    #[test]
    fn test_cluster_classifier_kmeans(){
        let train = "train";
        let train_images = load_data_set2(train).unwrap();
        assert_eq!(NUM_TRAINING, train_images.len());
        let images: Vec<Image>= train_images.iter().map(|(_,img)| img.clone()).collect();

        let kmeans = print_time_milliseconds("Kmeans++ clustering",
                                             || Kmeans::new(NUM_CENTROIDS, &images, euclidean_distance, image_mean_borrowed));
        println!();
        let means = kmeans.move_means();

        classify_label(&train_images, &means);
    }

    #[test]
    fn test_cluster_classifier_soc(){
        let train = "train";
        let train_images = load_data_set2(train).unwrap();
        let images: Vec<Image>= train_images.iter().map(|(_,img)| img.clone()).collect();

        let soc = print_time_milliseconds("SOC++ weighted clustering",
                                          || SOCluster::new_trained(NUM_CENTROIDS, &images, euclidean_distance));
        println!();
        let means = soc.move_clusters();

        classify_label(&train_images, &means);
    }

    #[test]
    fn test_cluster_classifier_soc_unweighted(){
        let train = "train";
        let train_images = load_data_set2(train).unwrap();

        let images: Vec<Image>= train_images.iter().map(|(_,img)| img.clone()).collect();

        let soc = print_time_milliseconds("SOC++ unweighted clustering",
                                          || SOCluster::new_trained_unweighted(NUM_CENTROIDS, &images, euclidean_distance));
        println!();
        let means = soc.move_clusters();

        classify_label(&train_images, &means);
    }

    fn get_clustering_counts(name: &str, data: &Vec<Image>, means: &Vec<Image>){
        let mut counts: Vec<usize> = (0..means.len()).map(|_| 0).collect();
        for img in data{
            let closest: (f64, usize) = classify(img, &means, &euclidean_distance);
            counts[closest.1] += 1;
        }

        println!();
        println!("{} Counts: {:?}", name, counts);

    }

    fn classify_label(train_images: &Vec<(u8, Image)>, means: &Vec<Image>){
        let label_histogram = label_classifier(train_images, &means);
        assert_eq!(means.len(), label_histogram.len());

        let test = "t10k";
        let test_images = load_data_set2(test).unwrap();


        let mut results: HashHistogram<bool> = HashHistogram::new();

        for (label, img) in test_images{
            let closest: (f64, usize) = classify(&img, &means, &euclidean_distance);
            let guessed_label = label_histogram[closest.1].mode();
            let mut iscorrect: bool;
            if guessed_label.eq(&label){
                iscorrect = true;
            }else {
                iscorrect = false;
            }
            results.bump(iscorrect);


        }
        println!();
        println!("Test Results: ");
        println!("correct: {}, incorrect: {}", results.get(true), results.get(false));

    }

    fn label_classifier(images: &Vec<(u8, Image)>, means: &Vec<Image>) -> Vec<HashHistogram<u8>> {
        let mut label_counter: Vec<HashHistogram<u8>> = (0..means.len()).map(|_| HashHistogram::new()).collect();
        for (label, img) in images {
            let closest: (f64, usize) = classify(img, means, &euclidean_distance);
            label_counter[closest.1].bump(label.clone());
        }
        label_counter

    }

}