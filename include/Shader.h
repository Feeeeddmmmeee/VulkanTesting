#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include <fstream>

struct VulkanShader
{
	vk::raii::ShaderModule module=nullptr;

	VulkanShader(const vk::raii::Device &device, const std::string &path)
	{
		auto shaderBin = readFile(path);
		vk::ShaderModuleCreateInfo info{
			.codeSize = shaderBin.size() * sizeof(char),
				.pCode = (uint32_t*)shaderBin.data()
		};
		module = vk::raii::ShaderModule{device, info};
	}

	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if(!file.is_open())
			throw std::runtime_error("Failed to open file!");

		std::vector<char> buffer(file.tellg()); // ate => we start at the end of file and thus we know its size
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), buffer.size());
		file.close();
		return buffer;
	}
};

