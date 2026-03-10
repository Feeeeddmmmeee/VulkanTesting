#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include <unordered_map>

struct PipelineKey
{
	std::string vertMain;
	std::string fragMain;
	std::string vert;
	std::string frag;

	bool operator==(const PipelineKey&) const = default;
};

namespace std {
    template<> struct hash<PipelineKey> {
        size_t operator()(PipelineKey const& pipeline) const {
			size_t hsh = 0;

			auto combine = [&hsh](const std::string &s) {
				return (hsh ^ (hash<string>{}(s) << 1)) >> 1;
			};

			combine(pipeline.vert);
			combine(pipeline.frag);
			combine(pipeline.vertMain);
			combine(pipeline.fragMain);
			return hsh;
        }
    };
}

struct VulkanPipeline
{
	vk::raii::Pipeline pipeline = nullptr;
	vk::raii::PipelineLayout layout = nullptr;

	VulkanPipeline(const vk::raii::Device &dev, const PipelineKey &key, const vk::SurfaceFormatKHR &surfaceFormat, vk::Format depthFormat, const vk::raii::DescriptorSetLayout &descSetLayout, vk::SampleCountFlagBits msaaSamples);
};

class PipelineManager
{
	public:
		PipelineManager(const vk::raii::Device &dev, const vk::raii::DescriptorSetLayout &layout, vk::Format depth, const vk::SurfaceFormatKHR &surface, vk::SampleCountFlagBits msaaSamples) :
			device(dev), descSetLayout(layout), depthFormat(depth), surfaceFormat(surface),
			msaaSamples(msaaSamples) {}
		std::shared_ptr<VulkanPipeline> get(const PipelineKey &key);

	private:
		vk::Format depthFormat;
		vk::SampleCountFlagBits msaaSamples;
		const vk::SurfaceFormatKHR &surfaceFormat;
		const vk::raii::DescriptorSetLayout &descSetLayout;
		const vk::raii::Device &device;

		std::unordered_map<PipelineKey, std::shared_ptr<VulkanPipeline>> cache;

		std::shared_ptr<VulkanPipeline> createPipeline(const PipelineKey &key);
};
