#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include "Pipeline.h"

struct Material
{
	std::shared_ptr<VulkanPipeline> pipeline = nullptr;

	void setPipeline(std::shared_ptr<VulkanPipeline> pipeline)
	{
		this->pipeline = pipeline;
	}
};

struct Mesh
{
		vk::raii::Buffer vertexBuffer = nullptr;
		vk::raii::DeviceMemory vBufferMemory = nullptr;
		vk::raii::Buffer indexBuffer = nullptr;
		vk::raii::DeviceMemory iBufferMemory = nullptr;

		uint32_t vertexCount = 0;
		Material material;
};

struct Model
{
	std::vector<Mesh> meshes;
};
