/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub fn risc_is_feature_supported(feature: &str) -> bool {
    let path = Path::new("/proc/cpuinfo");
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(_) => {
            return false;
        }
    };

    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => {
                return false;
            }
        };

        // Look for the line that starts with "isa"
        if line.starts_with("isa") || line.starts_with("ISA") {
            if line.contains(feature) {
                return true;
            } else {
                println!("{} extension is not supported", feature);
            }
            break; // Exit the loop once we've found and checked the ISA line
        }
    }

    false
}

pub fn risc_is_features_supported(feature: &[String]) -> bool {
    let path = Path::new("/proc/cpuinfo");
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(_) => {
            return false;
        }
    };

    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => {
                return false;
            }
        };

        // Look for the line that starts with "isa"
        if line.starts_with("isa") || line.starts_with("ISA") {
            for ft in feature.iter() {
                if !line.contains(ft) {
                    return false;
                }
            }
            return true;
        }
    }

    false
}
