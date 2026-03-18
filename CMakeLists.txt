cmake_minimum_required(VERSION 3.14)
project(Semat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(semat src/semat.cc)
target_link_libraries(semat Threads::Threads)
