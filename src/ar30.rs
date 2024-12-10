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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Rgb30 {
    Ar30 = 0,
    Ra30 = 1,
}

impl From<usize> for Rgb30 {
    fn from(value: usize) -> Self {
        match value {
            0 => Rgb30::Ar30,
            1 => Rgb30::Ra30,
            _ => {
                unimplemented!("Rgb30 is not implemented for value {}", value)
            }
        }
    }
}

/// Converts a value from host byte order to network byte order.
#[inline]
const fn htonl(hostlong: u32) -> u32 {
    hostlong.to_be()
}

/// Converts a value from network byte order to host byte order.
#[inline]
const fn ntohl(netlong: u32) -> u32 {
    u32::from_be(netlong)
}

impl Rgb30 {
    #[inline]
    pub(crate) const fn pack_w_a<const STORE: usize>(self, r: i32, g: i32, b: i32, a: i32) -> u32 {
        let value: u32 = match self {
            Rgb30::Ar30 => (a << 30 | (b << 20) | (g << 10) | r) as u32,
            Rgb30::Ra30 => ((r << 22) | (g << 12) | (b << 2) | a) as u32,
        };
        if STORE == 0 {
            value
        } else {
            htonl(value)
        }
    }

    #[inline(always)]
    pub(crate) const fn unpack<const STORE: usize>(self, value: u32) -> (u32, u32, u32, u32) {
        let pixel = if STORE == 0 { value } else { ntohl(value) };
        match self {
            Rgb30::Ar30 => {
                let r10 = pixel & 0x3ff;
                let g10 = (pixel >> 10) & 0x3ff;
                let b10 = (pixel >> 20) & 0x3ff;
                let a10 = pixel >> 30;
                (r10, g10, b10, a10)
            }
            Rgb30::Ra30 => {
                let a2 = pixel & 0x3;
                let r10 = (pixel >> 22) & 0x3ff;
                let g10 = (pixel >> 12) & 0x3ff;
                let b10 = (pixel >> 2) & 0x3ff;
                (r10, g10, b10, a2)
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Defines storage byte order for RGBA1010102 or RGBA2101010
///
/// Some systems require to be bytes in network byte order instead of host.
pub enum Ar30ByteOrder {
    Host = 0,
    Network = 1,
}

impl From<usize> for Ar30ByteOrder {
    fn from(value: usize) -> Self {
        match value {
            0 => Ar30ByteOrder::Host,
            1 => Ar30ByteOrder::Network,
            _ => {
                unimplemented!("Rgb30ByteOrder is not implemented for value {}", value)
            }
        }
    }
}
