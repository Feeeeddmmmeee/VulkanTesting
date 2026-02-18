cmake_minimum_required(VERSION 3.21)

project(VulkanTesting)
set(CMAKE_CXX_STANDARD 20)

# LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(_DEBUG)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

find_package(slang CONFIG HINTS "$ENV{VULKAN_SDK}/lib/make")
find_package (Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries (${PROJECT_NAME} Vulkan::Vulkan glm::glm SDL3::SDL3 glfw)
