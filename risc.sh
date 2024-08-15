#
# Copyright (c) Radzivon Bartoshyk. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1.  Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3.  Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

qemu-system-riscv64 \
    -machine virt \
    -cpu rv64,v=true \
    -m 4G \
    -nographic \
    -bios default \
    -hda /Users/radzivon/qemu/ubuntu-latest.qcow2 \
    -drive file=/Users/radzivon/Downloads/ubuntu-24.04-preinstalled-server-riscv64.img,format=raw,if=virtio \
    -bios /Users/radzivon/Downloads/fw_jump.elf \
    -kernel /Users/radzivon/Downloads/uboot.elf \
    -device virtio-net-device,netdev=net0 \
    -device virtio-rng-pci \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -serial mon:stdio

qemu-system-riscv64 \
    -machine virt \
    -cpu rv64,zba=true,zbb=true,v=true,vlen=256,vext_spec=v1.0,rvv_ta_all_1s=true,rvv_ma_all_1s=true \
    -m 4G \
    -nographic \
    -bios default \
    -drive file=/Users/radzivon/qemu/ubuntu-latest.qcow2,if=virtio \
    -bios /Users/radzivon/Downloads/fw_jump.elf \
    -kernel /Users/radzivon/Downloads/uboot.elf \
    -device virtio-net-device,netdev=net0 \
    -device virtio-rng-pci \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -serial mon:stdio