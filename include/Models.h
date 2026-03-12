#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include "Pipeline.h"
#include "Texture.h"
#include "Buffer.h"

struct Material
{
	std::shared_ptr<VulkanPipeline> pipeline = nullptr;
	std::shared_ptr<Texture> texture = nullptr;

	std::vector<vk::raii::DescriptorSet> descriptorSets;

	void setPipeline(std::shared_ptr<VulkanPipeline> pipeline)
	{
		this->pipeline = pipeline;
	}
};

struct Mesh
{
	std::shared_ptr<VulkanBuffer> vertexBuffer = nullptr;
	std::shared_ptr<VulkanBuffer> indexBuffer = nullptr;

	uint32_t vertexCount = 0;
	Material material;
};

struct Model
{
	std::vector<Mesh> meshes;
};
