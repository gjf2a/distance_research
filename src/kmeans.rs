use decorum::R64;
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};

#[allow(unused_imports)]
use decorum::Real; // TODO: Report bug. Not needed to compile. Solely to satisfy the IntelliJ type checker.

#[allow(dead_code)] // distance is only used in the test code, for now, as it is used strictly as a parameter during initialization.
pub struct Kmeans<T, V: Copy + Eq + Ord, D: Fn(&T,&T) -> V> {
    means: Vec<T>,
    distance: D
}

impl <T: Clone + Eq, V: Copy + Eq + Ord + Into<f64>, D: Fn(&T,&T) -> V> Kmeans<T,V,D> {
    pub fn new<M: Fn(&Vec<T>) -> T>(k: usize, data: &[T], distance: D, mean: M) -> Kmeans<T,V,D> {
        Kmeans {means: kmeans_iterate(k, data, &distance, &mean), distance}
    }

    #[cfg(test)]
    pub fn k(&self) -> usize {self.means.len()}

    #[cfg(test)]
    pub fn classification(&self, sample: &T) -> usize {
        classify(sample, &self.means, &self.distance)
    }

    #[cfg(test)]
    pub fn copy_means(&self) -> Vec<T> {self.means.clone()}

    pub fn move_means(self) -> Vec<T> {self.means}
}

fn initial_plus_plus<T: Clone + Eq, V: Copy + Eq + Ord + Into<f64>, D: Fn(&T,&T) -> V>(k: usize, distance: &D, data: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    let mut rng = thread_rng();
    let range = Uniform::new(0, data.len());
    result.push(data[range.sample(&mut rng)].clone());
    while result.len() < k {
        let squared_distances: Vec<f64> = data.iter()
            .map(|datum| 1.0 + distance(datum, result.last().unwrap()).into().powf(2.0))
            .collect();
        let dist = WeightedIndex::new(&squared_distances).unwrap();
        result.push(data[dist.sample(&mut rng)].clone());
    }
    result
}

fn kmeans_iterate<T: Clone + Eq, V: Copy + Eq + Ord + Into<f64>, D: Fn(&T,&T) -> V, M: Fn(&Vec<T>) -> T>(k: usize, data: &[T], distance: &D, mean: &M) -> Vec<T> {
    let mut result = initial_plus_plus(k, distance, data);
    loop {
        let mut classifications: Vec<Vec<T>> = (0..k).map(|_| Vec::new()).collect();
        for datum in data {
            let category = classify(datum, &result, distance);
            classifications[category].push(datum.clone());
        }
        let prev = result;
        result = (0..k)
            .map(|i|
                if !classifications[i].is_empty() {
                    mean(&classifications[i])
                } else {
                    prev[i].clone()
                })
            .collect();

        if (0..result.len()).all(|i| prev[i] == result[i]) {
            return result;
        }
    }
}

fn classify<T: Clone + Eq, V: Copy + Eq + Ord + Into<f64>, D: Fn(&T,&T) -> V>(target: &T, means: &Vec<T>, distance: &D) -> usize {
    let distances: Vec<(R64,usize)> = (0..means.len())
        .map(|i| (R64::from_inner(distance(&target, &means[i]).into()), i))
        .collect();
    distances.iter().min().unwrap().1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manhattan(n1: &i32, n2: &i32) -> R64 {
        let mut diff = n1 - n2;
        if diff < 0 {diff = -diff;}
        R64::from_inner(diff as f64)
    }

    fn mean(nums: &Vec<i32>) -> i32 {
        let total: i32 = nums.iter().sum();
        total / (nums.len() as i32)
    }

    #[test]
    fn test_k_means() {
        let target_means = vec![3, 11, 25, 40];
        let data = vec![2, 3, 4, 10, 11, 12, 24, 25, 26, 35, 40, 45];
        let kmeans =
            Kmeans::new(target_means.len(), &data, manhattan, mean);
        let mut sorted_means = kmeans.copy_means();
        sorted_means.sort();
        let unsorted_means = kmeans.copy_means();
        assert_eq!(kmeans.k(), sorted_means.len());
        assert_eq!(sorted_means.len(), target_means.len());
        for i in 0..sorted_means.len() {
            assert_eq!(sorted_means[i], target_means[i]);
            let matching_mean = unsorted_means[kmeans.classification(&target_means[i])];
            let sorted_index = sorted_means.binary_search(&matching_mean).unwrap();
            assert_eq!(i, sorted_index);
        }
    }
}