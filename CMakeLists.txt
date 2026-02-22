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
find_package (glm REQUIRED)

target_link_libraries (${PROJECT_NAME} PRIVATE Vulkan::Vulkan glm::glm)

set(WINDOW_BACKEND "SDL" CACHE STRING "Choose the window creation library (SDL/GLFW)")

message(STATUS "Using window backend: ${WINDOW_BACKEND}")
if(WINDOW_BACKEND STREQUAL "SDL")
	find_package(SDL3 REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)
	add_compile_definitions(SDL)

elseif(WINDOW_BACKEND STREQUAL "GLFW")
	find_package(glfw3 REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
	add_compile_definitions(GLFW)

else()
	message(FATAL_ERROR "Unknown WINDOW_BACKEND: '${WINDOW_BACKEND}'! Must be one of: SDL, GLFW")

endif()
