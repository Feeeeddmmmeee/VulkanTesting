#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

struct Texture
{
	uint32_t mipLevels;
	vk::raii::Image textureImage = nullptr;
	vk::raii::ImageView textureImageView = nullptr;
	vk::raii::DeviceMemory textureMemory = nullptr;
	vk::raii::Sampler textureSampler = nullptr;
};
