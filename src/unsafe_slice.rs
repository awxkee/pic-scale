/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::cell::UnsafeCell;
use std::ops::Index;

#[derive(Copy, Clone)]
pub struct UnsafeSlice<'a, T> {
    pub slice: &'a [UnsafeCell<T>],
}

unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}

unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    pub fn mut_ptr(&self) -> *mut T {
        self.slice.as_ptr() as *const T as *mut T
    }

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    #[allow(dead_code)]
    pub unsafe fn write(&self, i: usize, value: T) {
        let ptr = self.slice[i].get();
        *ptr = value;
    }
    #[allow(dead_code)]
    pub fn get(&self, i: usize) -> &T {
        let ptr = self.slice[i].get();
        unsafe { &*ptr }
    }
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.slice.len()
    }
}

impl<'a, T> Index<usize> for UnsafeSlice<'a, T> {
    type Output = T;
    #[allow(dead_code)]
    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self.slice[index].get();
        unsafe { &*ptr }
    }
}
