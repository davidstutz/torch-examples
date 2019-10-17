# Torch C/CUDA Module Template

This repository contains a template for extending Torch with custom
C/CUDA modules without requiring knowledge of the Torch core. The
template is discussed in detail in [this article](TODO).

## Requirements

It is recommended to have Torch installed through [torch/distro](https://github.com/torch/distro)
which includes all necessary Torch modules. In addition, LUA's [ffi package](http://luajit.org/ext_ffi.html)
is required.

For compiling the C/CUDA modules, CMake and CUDA are required. The code
has been tested using compute capability `sm_20`. Depending on your graphics card
and CUDA version, `lib/cpp/gpu/CMakeLists.txt` needs to be adapted; specifically
check for the `-arch=sm_20` option.

## Installation and Test

The C/CUDA modules are built using:

    cd lib/cpp/cpu
    mkdir build
    cd build
    cmake ..
    make
    cd ..
    cd gpu
    mkdir build
    cd build
    cmake ..
    make
    cd ../../../../
    th test.lua

## License

Copyright (c) 2018 David Stutz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.