cmake_minimum_required(VERSION 3.21)

project(VulkanTesting)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")

# LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(_DEBUG)

# file(GLOB_RECURSE SRC_FILES src/*.cpp)
set(SRC_FILES src/main.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

find_package(slang CONFIG HINTS "$ENV{VULKAN_SDK}/lib/make")
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glm::glm)
target_include_directories(${PROJECT_NAME} PRIVATE include/ src/)

set(WINDOW_BACKEND "SDL" CACHE STRING "Choose the window creation library (SDL/GLFW)")

message(STATUS "Using window backend: ${WINDOW_BACKEND}")
if(WINDOW_BACKEND STREQUAL "SDL")
	find_package(SDL3 REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)
	file(GLOB_RECURSE SDL_SRC src/Platform/SDL/*.cpp)
	target_sources(${PROJECT_NAME} PRIVATE ${SDL_SRC})
	add_compile_definitions(SDL)

elseif(WINDOW_BACKEND STREQUAL "GLFW")
	find_package(glfw3 REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
	file(GLOB_RECURSE GLFW_SRC src/Platform/GLFW/*.cpp)
	target_sources(${PROJECT_NAME} PRIVATE ${GLFW_SRC})
	add_compile_definitions(GLFW)

else()
	message(FATAL_ERROR "Unknown WINDOW_BACKEND: '${WINDOW_BACKEND}'! Must be one of: SDL, GLFW")

endif()
