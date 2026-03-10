#include "Pipeline.h"
#include "Shader.h"
#include "Vertex.h"

VulkanPipeline::VulkanPipeline(const vk::raii::Device &dev, const PipelineKey &key, const vk::SurfaceFormatKHR &surfaceFormat, vk::Format depthFormat, const vk::raii::DescriptorSetLayout &descSetLayout, vk::SampleCountFlagBits msaaSamples)
{
	// modules should probably be cached at some point

	VulkanShader vert(dev, key.vert);
	VulkanShader frag(dev, key.frag);
	vk::PipelineShaderStageCreateInfo vertInfo{
		.stage = vk::ShaderStageFlagBits::eVertex,
			.module = vert.module,
			.pName = key.vertMain.c_str()
	};
	vk::PipelineShaderStageCreateInfo fragInfo{
		.stage = vk::ShaderStageFlagBits::eFragment,
			.module = frag.module,
			.pName = key.fragMain.c_str()
	};
	vk::PipelineShaderStageCreateInfo shaderStages[] = {vertInfo, fragInfo};

	auto bindingDescription = Vertex::getBindingDesc();
	auto attrDescriptions = Vertex::getAttributeDescs();
	vk::PipelineVertexInputStateCreateInfo vertInputInfo{
		.vertexBindingDescriptionCount=1,
			.pVertexBindingDescriptions=&bindingDescription,
			.vertexAttributeDescriptionCount=attrDescriptions.size(),
			.pVertexAttributeDescriptions=attrDescriptions.data()
	};

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology=vk::PrimitiveTopology::eTriangleList};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};
	std::vector dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};
	vk::PipelineDynamicStateCreateInfo dynamicState{
		.dynamicStateCount = (uint32_t)dynamicStates.size(),
			.pDynamicStates = dynamicStates.data()
	};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.depthBiasSlopeFactor = 1.0f,
			.lineWidth = 1.0f
	};

	vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples=msaaSamples,.sampleShadingEnable=vk::False};
	vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable    = vk::True,
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	// Alpha blending
	colorBlendAttachment.blendEnable = vk::True;
	colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
	colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
	colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
	colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
	colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
	colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
	vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp =  vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments =  &colorBlendAttachment };

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount=1,
		.pSetLayouts=&*descSetLayout,			
		.pushConstantRangeCount=0
	};

	layout = vk::raii::PipelineLayout(dev, pipelineLayoutInfo);

	vk::PipelineRenderingCreateInfo renderingCreateInfo{
		.colorAttachmentCount=1,
			.pColorAttachmentFormats=&surfaceFormat.format,
			.depthAttachmentFormat=depthFormat
	};

	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable       = vk::True,
			.depthWriteEnable      = vk::True,
			.depthCompareOp        = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable     = vk::False
	};

	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.pNext = &renderingCreateInfo,
			.stageCount = 2, .pStages = shaderStages,
			.pVertexInputState = &vertInputInfo, .pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState, .pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil, .pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState, .layout = layout, 
			.renderPass = nullptr,
	};

	pipeline = vk::raii::Pipeline(dev, nullptr, pipelineInfo);
}

std::shared_ptr<VulkanPipeline> PipelineManager::get(const PipelineKey &key)
{
	auto it = this->cache.find(key);
	if(it != this->cache.end())
		return it->second;

	auto pipeline = std::make_shared<VulkanPipeline>(this->device, key, surfaceFormat, depthFormat, descSetLayout, msaaSamples);
	this->cache[key] = pipeline;
	return pipeline;
}
