#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

// todo: image, buffer (copy requries a command buffer)

struct VulkanImage
{
	vk::raii::Image image = nullptr;
	vk::raii::ImageView view = nullptr;
	vk::raii::DeviceMemory memory = nullptr;

	uint32_t mipLevels;
	void createImageView(vk::Format format, vk::ImageAspectFlags aspect, vk::raii::Device &device);
};

struct Texture
{
	Texture(vk::raii::Device &dev, vk::raii::PhysicalDevice &pdev) :
		device(dev), pDevice(pdev) {}

	VulkanImage image;
	vk::raii::Sampler sampler = nullptr;

	vk::raii::PhysicalDevice &pDevice;
	vk::raii::Device &device;

	void createSampler();
	void createImageView();
};
