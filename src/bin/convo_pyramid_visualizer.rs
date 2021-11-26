use iced::{Element, Sandbox};
use distance_research::mnist_data::{Image, load_data_set};

fn main() {

}

struct Visualizer {
    images: Vec<(u8, Image)>,
    current: usize
}

impl Sandbox for Visualizer {
    type Message = ();

    fn new() -> Self {
        todo!()
    }

    fn title(&self) -> String {
        todo!()
    }

    fn update(&mut self, message: Self::Message) {
        todo!()
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        todo!()
    }
}