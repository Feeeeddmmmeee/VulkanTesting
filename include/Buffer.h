#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

struct VulkanBuffer
{
	vk::raii::Buffer buffer = nullptr;
	vk::raii::DeviceMemory memory = nullptr;
	vk::DeviceSize size;
	void *data = nullptr;

	vk::raii::Device &device;
	vk::raii::PhysicalDevice &pDevice;

	VulkanBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
			vk::MemoryPropertyFlags memoryFlags, vk::raii::Device &dev,
			vk::raii::PhysicalDevice &pDev
		);

	void mapMemory();
	void unmapMemory();
	void uploadToMemory(void *src, int offset=0);
};
