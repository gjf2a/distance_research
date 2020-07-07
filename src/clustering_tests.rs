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

fn max_f64(a: f64, b: f64) -> f64 {
    if a < b {
        return b;
    }
    return a;
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b {
        return a;
    }
    return b;
}

pub mod tests{
    use super::*;
    use soc::*;
    use kmeans::*;
    use std::io;
    use crate::mnist_data::{Image, init_from_files, image_mean, Grid};
    use crate::timing::print_time_milliseconds;
    use crate::euclidean_distance::euclidean_distance;
    use hash_histogram::HashHistogram;
    use std::collections::BTreeSet;
    use std::cmp::{max, min};
    use crate::permutation::make_permutation;
    use itertools::Itertools;

    const BASE_PATH: &str = "/Users/davidpojunas/Desktop/Y3S2/distance_research/distance_research/mnist_data2/";
    const NUM_CENTROIDS: usize = 8;
    const TAKE: usize = 1000;
    const NUM_TRAINING: usize = 60000;

    fn load_data_set2(file_prefix: &str) -> io::Result<Vec<(u8,Image)>> {
        let train_images = format!("{}{}-images-idx3-ubyte", BASE_PATH, file_prefix);
        let train_labels = format!("{}{}-labels-idx1-ubyte", BASE_PATH, file_prefix);

        let training_images = print_time_milliseconds(&format!("loading mnist {} images", file_prefix),
                                                      || init_from_files(train_images.as_str(), train_labels.as_str()))?;

        println!("Number of {} images: {}", file_prefix, training_images.len());
        Ok(training_images)
    }

    #[test]
    fn test_kmeans_plus_plus(){
        let train = "train";
        //let test = "t10k";
        let train_images = load_data_set2(train).unwrap();
        //let testing_images = load_data_set2(test).unwrap();
        assert_eq!(NUM_TRAINING, train_images.len());
        let perm = make_permutation(28*28);
        //assert_eq!(10000, testing_images.len());
        let images: Vec<Image> = train_images.iter().map(|(_,img)| img.permuted(&perm).clone()).collect();
        let means = Kmeans::new(NUM_CENTROIDS, &images, euclidean_distance, image_mean);
        let centroids = means.move_means();
        for (i, img) in centroids.iter().enumerate() {
            let name = format!("{}", i);
            img.save(name);
        }
        let mut average: f64 = 0.0;
        let mut max1 = f64::MIN;
        let mut min1 = f64::MAX;
        for img in images {
            let (dist, index): (f64, usize) = classify(&img, &centroids, &euclidean_distance);
            average += dist;
            max1 = max_f64(dist, max1);
            min1 = min_f64(dist, min1);
        }
        average = average / (NUM_TRAINING as f64);
        println!();
        println!("Average: {}, Max: {}, Min: {}", average, max1, min1);

    }
    //Kmeans++
    //1000 images--Average: 2534580.095, Max: 5960902, Min: 719901
    //all images--Average: 2639960.968, Max: 8701182, Min: 529856
    //permuted all images--Average: 2639958.1052, Max: 8695826, Min: 531645

    //SOC
    //1000 images-- 3720567.967, Max: 7370547, Min: 0
    //all images--Average: 4340943.188, Max: 9418408, Min: 0
    //permuted all images--Average: 4340943.188, Max: 9418408, Min: 0



    #[test]
    fn test_soc_original(){
        let train = "train";
        let train_images = load_data_set2(train).unwrap();
        assert_eq!(NUM_TRAINING, train_images.len());
        let images: Vec<Image> = train_images.iter().map(|(_,img)| img.clone()).collect();
        let means = SOCluster::new_trained(NUM_CENTROIDS, &images, euclidean_distance, image_mean);
        let centroids = means.move_clusters();
        for (i, img) in centroids.iter().enumerate() {
            let name = format!("{}", i);
            img.save(name);
        }
        let mut average: f64 = 0.0;
        let mut max1 = f64::MIN;
        let mut min1 = f64::MAX;
        for img in images {
            let (dist, index): (f64, usize) = classify(&img, &centroids, &euclidean_distance);
            average += dist;
            max1 = max_f64(dist, max1);
            min1 = min_f64(dist, min1);
        }
        average = average / (NUM_TRAINING as f64);
        println!();
        println!("Average: {}, Max: {}, Min: {}", average, max1, min1);

    }


    #[test]
    fn test_soc_kmeans(){
        let train = "train";
        let train_images = load_data_set2(train).unwrap();
        assert_eq!(NUM_TRAINING, train_images.len());
        let images: Vec<Image> = train_images.iter().map(|(_,img)| img.clone()).collect();

        let kmeans =  Kmeans::new(NUM_CENTROIDS, &images, euclidean_distance, image_mean);
        let kmeans_means: Vec<Image> = kmeans.move_means();


        let soc = SOCluster::new_trained(NUM_CENTROIDS, &images, euclidean_distance, image_mean);
        let soc_means: Vec<Image> = soc.move_clusters();

        assert_eq!(kmeans_means.len(), soc_means.len());
        assert_eq!(NUM_CENTROIDS, soc_means.len());


        let mut kmeans_clustered: Vec<Vec<f64>> = (0..NUM_CENTROIDS).map(|_| Vec::new()).collect();
        let mut soc_clustered: Vec<Vec<f64>> = (0..NUM_CENTROIDS).map(|_| Vec::new()).collect();;

        for img in &images {
            let kmeans_closest: (f64, usize) = classify(img, &kmeans_means, &euclidean_distance);
            let soc_closest: (f64, usize) = classify(img, &soc_means, &euclidean_distance);
            kmeans_clustered[kmeans_closest.1].push(kmeans_closest.0);
            soc_clustered[soc_closest.1].push(soc_closest.0);
        }

        assert_eq!(kmeans_clustered.len(), soc_clustered.len());

        let mut kmeans_dist_mean: Vec<f64> = Vec::new();
        let mut soc_dist_mean: Vec<f64> = Vec::new();

        for (i, vec) in kmeans_clustered.iter().enumerate(){
            let k_sum: f64 = vec.iter().sum();
            let k_mean: f64 = k_sum / (vec.len() as f64);
            kmeans_dist_mean.push(k_mean);
            println!("kmeans cluster {} mean distance: {}", i, k_mean);

            let soc_sum: f64 = soc_clustered[i].iter().sum();
            let soc_mean: f64 = soc_sum / (soc_clustered.len() as f64);
            soc_dist_mean.push(soc_mean);
            println!("soc cluster {} mean distance: {}", i, soc_mean);
        }

        assert_eq!(kmeans_dist_mean.len(), soc_dist_mean.len());

        println!();

        let kmeans_average_mean: f64 = kmeans_dist_mean.iter().sum::<f64>() / (kmeans_dist_mean.len() as f64);
        let soc_average_mean: f64 = soc_dist_mean.iter().sum::<f64>() / (soc_dist_mean.len() as f64);
        println!("kmeans all clusters average mean distance: {}", kmeans_average_mean);
        println!("soc all clusters average mean distance: {}", soc_average_mean);

        let mut kmeans_dist_var: Vec<f64> = Vec::new();
        let mut soc_dist_var: Vec<f64> = Vec::new();

        println!();

        for (i, vec) in kmeans_clustered.iter().enumerate() {
            let k_v_sum = vec.iter().fold(0.0, |sum, val| sum + ((val - kmeans_dist_mean[i]).powf(2.0)));
            let k_variance = k_v_sum / (vec.len() as f64);
            kmeans_dist_var.push(k_variance);
            println!("kmeans cluster {} variance: {}", i, k_variance);

            let soc_v_sum = soc_clustered[i].iter().fold(0.0, |sum, val| sum + ((val - soc_dist_mean[i]).powf(2.0)));
            let soc_variance = soc_v_sum / (soc_clustered.len() as f64);
            soc_dist_var.push(soc_variance);
            println!("soc cluster {} variance: {}", i, soc_variance);
        }

        assert_eq!(kmeans_dist_var.len(), soc_dist_var.len());

        println!();

        let kmeans_average_var: f64 = kmeans_dist_var.iter().sum::<f64>() / (kmeans_dist_var.len() as f64);
        let soc_average_var: f64 = soc_dist_var.iter().sum::<f64>() / (soc_dist_var.len() as f64);
        println!("kmeans all clusters average variance distance: {}", kmeans_average_var);
        println!("soc all clusters average variance distance: {}", soc_average_var);
    }

    #[test]
    fn test_soc_plus_plus(){
        let train = "train";
        let train_images = load_data_set2(train).unwrap();
        assert_eq!(NUM_TRAINING, train_images.len());
        let images: Vec<Image> = train_images.iter().map(|(_,img)| img.clone()).collect();

        let soc = SOCluster::new_trained_plus(NUM_CENTROIDS, &images, euclidean_distance, image_mean);
        let soc_means: Vec<Image> = soc.move_clusters();

        assert_eq!(NUM_CENTROIDS, soc_means.len());

        let mut soc_clustered: Vec<Vec<f64>> = (0..NUM_CENTROIDS).map(|_| Vec::new()).collect();;

        for img in &images {
            let soc_closest: (f64, usize) = classify(img, &soc_means, &euclidean_distance);
            soc_clustered[soc_closest.1].push(soc_closest.0);
        }

        let mut soc_dist_mean: Vec<f64> = Vec::new();

        for (i, vec) in soc_clustered.iter().enumerate(){
            let soc_sum: f64 = vec.iter().sum();
            let soc_mean: f64 = soc_sum / (vec.len() as f64);
            soc_dist_mean.push(soc_mean);
            println!("soc cluster {} mean distance: {}", i, soc_mean);
        }

        println!();

        let soc_average_mean: f64 = soc_dist_mean.iter().sum::<f64>() / (soc_dist_mean.len() as f64);
        println!("soc all clusters average mean distance: {}", soc_average_mean);

        let mut soc_dist_var: Vec<f64> = Vec::new();

        println!();

        for (i, vec) in soc_clustered.iter().enumerate(){
            let soc_v_sum = vec.iter().fold(0.0, |sum, val| sum + ((val - soc_dist_mean[i]).powf(2.0)));
            let soc_variance = soc_v_sum / (soc_clustered.len() as f64);
            soc_dist_var.push(soc_variance);
            println!("soc cluster {} variance: {}", i, soc_variance);
        }

        println!();

        let soc_average_var: f64 = soc_dist_var.iter().sum::<f64>() / (soc_dist_var.len() as f64);
        println!("soc all clusters average variance distance: {}", soc_average_var);
    }
}

//kmeans cluster 0 mean distance: 3154279.147262305
//soc cluster 0 mean distance: 21681431240.25
//kmeans cluster 1 mean distance: 2785531.962296487
//soc cluster 1 mean distance: 70738907.125
//kmeans cluster 2 mean distance: 2708951.002484875
//soc cluster 2 mean distance: 2111134514.625
//kmeans cluster 3 mean distance: 2971957.8881817046
//soc cluster 3 mean distance: 749978153
//kmeans cluster 4 mean distance: 2325606.0367851965
//soc cluster 4 mean distance: 820750954.25
//kmeans cluster 5 mean distance: 2844041.6107226107
//soc cluster 5 mean distance: 6005002749.5
//kmeans cluster 6 mean distance: 3256791.815201803
//soc cluster 6 mean distance: 265185415.25
//kmeans cluster 7 mean distance: 1830991.262439207
//soc cluster 7 mean distance: 852851977.375
//
//kmeans all clusters average mean distance: 2734768.840671774
//soc all clusters average mean distance: 4069634238.921875
//
//kmeans cluster 0 variance: 635504549627.201
//soc cluster 0 variance: 2419172252061064000000000
//kmeans cluster 1 variance: 684453143004.4192
//soc cluster 1 variance: 51092598518837630
//kmeans cluster 2 variance: 633852896865.2379
//soc cluster 2 variance: 1608392449377129700000
//kmeans cluster 3 variance: 522104758984.86444
//soc cluster 3 variance: 79101083177293020000
//kmeans cluster 4 variance: 431447776467.2398
//soc cluster 4 variance: 102059904008219310000
//kmeans cluster 5 variance: 311631728200.02716
//soc cluster 5 variance: 50907813931773830000000
//kmeans cluster 6 variance: 485546235602.1347
//soc cluster 6 variance: 4115172056629701000
//kmeans cluster 7 variance: 494398641941.2687
//soc cluster 7 variance: 148657226324829630000
//
//kmeans all clusters average variance distance: 524867466336.5491
//soc all clusters average variance distance: 309002805365047600000000