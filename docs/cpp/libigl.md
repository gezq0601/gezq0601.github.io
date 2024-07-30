# 安装MinGW和Cmake

[msvc x86_64版本MinGW](https://github.com/niXman/mingw-builds-binaries/releases/download/13.2.0-rt_v11-rev0/x86_64-13.2.0-release-posix-seh-msvcrt-rt_v11-rev0.7z)

1. 下载后解压，将mingw64放置到新目录。
2. 在环境变量PATH中添加mingw64的路径，需要精确到其中的bin子目录（如D:\Cpp\mingw64\bin）

[Download CMake](https://cmake.org/download/)

下载后直接安装，有添加到环境变量时勾选即可，否则也需要同上进行配置。

# libigl example project

[libigl/libigl-example-project: A blank project example showing how to use libigl and cmake. (github.com)](https://github.com/libigl/libigl-example-project)

1. 首先clone该项目

```bash
git clone https://github.com/libigl/libigl-example-project.git
```

2. 可以按照官网中所述进行操作，但此处不采用

```bash
mkdir build
cd build
cmake ..
make
```

3. 将该项目用VSCode打开，运行下述命令

```cmake
cmake -B build
```

4. 点击生成该项目，选择打开生成的example.exe即可

![image-20240730171205447](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/image-20240730171205447.png)

libigl官网：[libigl](https://libigl.github.io/)
