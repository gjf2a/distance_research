use bare_metal_modulo::{MNum, ModNum};
use iced::{Element, Sandbox, Settings};
use distance_research::mnist_data::{Image, load_data_set};

fn main() -> iced::Result {
    println!("Loading images....");
    Visualizer::run(Settings::default())
}

struct Visualizer {
    images: Vec<(u8, Image)>,
    current: ModNum<usize>
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Left, Right, GoTo(usize)
}

impl Sandbox for Visualizer {
    type Message = Message;

    fn new() -> Self {
        let images = load_data_set("train").unwrap();
        let current = ModNum::new(0, images.len());
        Visualizer { images, current }
    }

    fn title(&self) -> String {
        "Visualizer".to_owned()
    }

    fn update(&mut self, message: Self::Message) {
        match message {
            Message::Left => self.current -= 1,
            Message::Right => self.current += 1,
            Message::GoTo(index) => self.current.replace(index)
        }
    }

    fn view(&mut self) -> Element<'_, Self::Message> {
        todo!()
    }
}