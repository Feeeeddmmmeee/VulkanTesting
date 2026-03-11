#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

// todo: image, buffer (copy requries a command buffer)

struct Texture
{
	Texture(vk::raii::Device &dev, vk::raii::PhysicalDevice &pdev) :
		device(dev), pDevice(pdev) {}

	uint32_t mipLevels;
	vk::raii::Image image = nullptr;
	vk::raii::ImageView imageView = nullptr;
	vk::raii::DeviceMemory memory = nullptr;
	vk::raii::Sampler sampler = nullptr;

	vk::raii::PhysicalDevice &pDevice;
	vk::raii::Device &device;

	void createSampler();
	void createImageView();
};
