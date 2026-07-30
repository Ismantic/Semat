cmake_minimum_required(VERSION 3.14)
project(Semat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(semat src/semat.cc)
target_link_libraries(semat Threads::Threads)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/smoke.py
                $<TARGET_FILE:semat> ${CMAKE_SOURCE_DIR}/scripts/process.py
                ${CMAKE_SOURCE_DIR}/scripts/print_topics.py
                ${CMAKE_SOURCE_DIR}/data/process.py
    )
endif()
