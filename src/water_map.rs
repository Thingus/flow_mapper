use ndarray::prelude::*;
use std::option;
use std::iter::FromIterator;
use std::iter::Sum;
use ndarray::ViewRepr;


const NORTH:(usize, usize) = (0,1);
const WEST:(usize, usize)  = (1,0);
const SOUTH:(usize, usize)  =(2,1);
const EAST:(usize, usize)  = (1,2);
const CENTRE:(usize, usize) = (1,1);

struct WaterMap<'l> {
    map: &'l Array2<i32>,
    flow_map: Array2<i32>,
    current: Array2<i32>,
    next: Array2<i32>
}

impl WaterMap<'_>{
    pub fn new(map_array : &Array2<i32>, start_point:(usize, usize)) -> WaterMap{
        let mut new = WaterMap{
            map: map_array,
            flow_map: build_flow_map(map_array),
            current: Array2::zeros(map_array.dim()),
            next: Array2::zeros(map_array.dim())
        };
        new.set_water(start_point);
        new.next.assign(&new.current);
        new
    }

    pub fn set_water(&mut self, start_point:(usize, usize)){
        self.current[[start_point.0, start_point.1]] = 1;
    }
}

impl Iterator for WaterMap<'_>{
    type Item = Array2<i32>;

    fn next(&mut self) -> Option<Array2<i32>> {
        let new_next = flow_step(&self.current, &self.flow_map);
        if self.current == new_next {
            None
        } else {
            self.current.assign(&self.next);
            self.next.assign(&new_next);
            let mut out = Array::zeros(self.current.raw_dim());
            out.assign(&self.current);
            Some(out)
        }
    }
}

fn functional_convolve(matrix:&Array2<i32>,
                       kern_func: fn(ArrayView2<i32>) -> i32,
                       boarder_value: i32)
                       -> Array2<i32>{
    let in_height = matrix.dim().0;
    let in_width = matrix.dim().1;
    let mut out = Array2::zeros((matrix.dim()));
    let mut boardered = Array2::from_elem((
        in_height + 2, in_width + 2),
        boarder_value
    );
    boardered.slice_mut(s![1..-1, 1..-1]).assign(matrix);

    for ((y,x),element) in matrix.indexed_iter() {
        let kernel_view = boardered.slice(s![y..y+3, x..x+3]);
        out[[y,x]] = kern_func(kernel_view);
    }
    out
}

fn flow_map_kernel(kern_elements:ArrayView2<i32>) -> i32 {
    let centre = kern_elements[[1,1]];
    let mut out = 0;
    if centre > kern_elements[WEST]{out += 8};
    if centre > kern_elements[NORTH]{out += 4};
    if centre > kern_elements[EAST]{out += 2};
    if centre > kern_elements[SOUTH]{out += 1};
    //print!{"{:?}: {:?}\n", kern_elements, out}
    out
}

fn build_flow_map(elev_map: &Array2<i32>) -> Array2<i32>{
    functional_convolve(
        elev_map,
        flow_map_kernel,
        9999)
}

fn flow_step(current_map:&Array2<i32>, flow_map:&Array2<i32>) -> Array2<i32> {
    let mut masked_flow_map:Array2<i32> = current_map * flow_map;
    functional_convolve(&masked_flow_map,
                        flow_step_kernel,
                        0)
}

fn flow_step_kernel(masked_map: ArrayView2<i32>) -> i32 {
    if masked_map[CENTRE] > 0 {
        1
    } else if ((8 & masked_map[EAST]) == 8)
        | ((4 & masked_map[SOUTH]) == 4)
        | ((2 & masked_map[WEST]) == 2)
        | ((1 & masked_map[NORTH]) == 1)
    {
        1
    } else {
        0
    }
}

fn inf_boarder(input: Option<&i32>) -> i32 {
    match input {
        Some(input) => *input,
        None             => 9999 as i32
    }
}

fn zero_boarder(input: Option<&i32>) -> i32 {
    match input{
        Some(input) => *input,
        None => 0 as i32
    }
}

fn simulate_flow(map : Array2<i32>, start_point:Vec<i32>) -> Array2<i32> {
    let mut water_is_present = Array2::zeros(map.dim());
    water_is_present[(start_point[0] as usize, start_point[1] as usize)] = 1;
    water_is_present
}

//#[test]
//fn load_elevation_map() {
//    let mut e = ElevationMap::new("test_map.tif");
//}

#[test]
fn test_basic_flow_simulation() {
        let map = array![
        [10,10,10,10,10],
        [10,10,10,10,10],
        [10,10,1,10,10],
        [10,10,10,10,10],
        [10,10,10,10,10]
    ];
    let target = array![
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ];
    let out = simulate_flow(map, vec![2,2]);
    assert_eq!(out, target)
}

#[test]
fn test_flow_simulation() {
    let map = array![
        [10,10,1,10,10],
        [10,10,2,10,10],
        [10,10,3,10,10],
        [10,10,2,10,10],
        [10,10,1,10,10]
    ];
    let target = array![
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ];
    let mut map = WaterMap::new(&map, (2,2));
    let out = map.last();
    assert_eq!(out, Some(target))
}

#[test]
fn test_funct_convolve() {
    let input = array![
        [2,3,4],
        [5,6,7],
        [8,9,10]
        ];
    let target = array![
        [16,27,20],
        [33,54,39],
        [28,45,32]
    ];
    let out = functional_convolve(&input,
                                  |a|a.sum(),
                                  0);
    assert_eq!(out, target)
}

#[test]
fn test_build_flow_map() {
    let input = array![
        [5, 5, 5, 5, 5],
        [5, 1, 6, 5, 5],
        [5, 2, 3, 4, 5],
        [5, 5, 5, 5, 5],
    ];
    let target = array![
        [0, 1, 0, 0, 0],
        [2, 0, 15,1, 0],
        [2, 4, 8, 8, 8],
        [0, 4, 4, 4, 0]
    ];
    let out = build_flow_map(&input);
    assert_eq!(out,target)
}

#[test]
fn test_flow_step() {
    let flow_map = array![
        [2,0,0,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [4,0,0,8,0]
    ];

    let current_step = array![
        [1,0,0,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [1,0,0,1,0]
    ];

    let next_step = array![
        [1,1,0,1,0],
        [0,0,0,1,0],
        [1,0,0,0,0],
        [1,0,1,1,0]
    ];
    let out = flow_step(&current_step, &flow_map);
    assert_eq!(out, next_step)
}

// #[test]
// fn test_flow_step_kernel() {
//     let kernel_input = vec![2,0,0,0,0];
//     assert_eq!(1, flow_step_kernel(kernel_input));
//     let kernel_input = vec![0,66,0,0,0];
//     assert_eq!(1, flow_step_kernel(kernel_input));
//     let kernel_input = vec![0,0,8,0,0];
//     assert_eq!(1, flow_step_kernel(kernel_input));
//     let kernel_input = vec![0,0,0,1,0];
//     assert_eq!(1, flow_step_kernel(kernel_input));
//     let kernel_input = vec![0,0,0,0,4];
//     assert_eq!(1, flow_step_kernel(kernel_input));
//
//     let kernel_input = vec![0,0,0,0,0];
//     assert_eq!(0, flow_step_kernel(kernel_input));
//     let kernel_input = vec![0,0,2,0,0];
//     assert_eq!(0, flow_step_kernel(kernel_input));
// }