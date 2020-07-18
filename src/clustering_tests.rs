use std::f64::NAN;

pub fn classify<T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V>(target: &T, means: &Vec<T>, distance: &D) -> (V, usize) {
    let distances: Vec<(V,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]).into(), i))
        .collect();
    let y = distances.iter()
        .fold(None, |m:Option<&(V, usize)>, d| m.map_or(Some(d), |m|
            Some(if m.0 < d.0 {m} else {d}))).unwrap();
    (y.0, y.1)
}



pub mod tests{
    use super::*;
    use soc::*;
    use kmeans::*;
    use std::io;
    use crate::mnist_data::{Image, init_from_files, image_mean, Grid, image_mean_borrowed};
    use crate::timing::print_time_milliseconds;
    use crate::euclidean_distance::euclidean_distance;
    use hash_histogram::HashHistogram;
    use std::collections::BTreeSet;
    use std::cmp::{max, min};
    use itertools::Itertools;

    const BASE_PATH: &str = "/Users/david/Desktop/RustProjects/distance_research/distance_research/mnist_data2/";
    const NUM_CENTROIDS: usize = 8;
    const NUM_TRAINING: usize = 60000;
    const NUM_TESTING: usize = 10000;

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
        let counts = soc.copy_counts();
        println!("soc++ unweighted counts: {:?}", counts);
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
        let counts = soc.copy_counts();
        println!("soc++ weighted counts: {:?}", counts);
        let means = soc.move_clusters();

        classify_label(&train_images, &means);
    }

    fn classify_label(train_images: &Vec<(u8, Image)>, means: &Vec<Image>){
        let label_histogram = label_classifier(train_images, &means);

        let test = "t10k";
        let test_images = load_data_set2(test).unwrap();
        assert_eq!(NUM_TESTING, test_images.len());

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