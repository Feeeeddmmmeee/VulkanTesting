#include <cstdint>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Window.h"

#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <array>

#define LOG(x) std::cout<<x<<std::endl;

#ifdef _DEBUG
constexpr bool _ENABLE_VALIDATION_LAYERS = true;
#else
constexpr bool _ENABLE_VALIDATION_LAYERS = false;
#endif

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
constexpr const char* SHADER_PATH = "shaders/vertexBuffer.spv";

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDesc()
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
	}
	 static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescs()
	 {
		 return {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))
		 };
	 }
};

const std::vector<Vertex> triangle = {
	{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

const std::vector<Vertex> square = {
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}
};

auto vertices = triangle;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName
};

class App
{
	public:
		void run()
		{
			initWindow();
			LOG("Window initialized...")
			initVulkan();
			LOG("Vulkan initialized...")
			mainLoop();
			LOG("Cleaning up...")
			cleanup();
			LOG("Exiting...")
		}

	private:
		Window *window;

		vk::raii::Context context;
		vk::raii::Instance instance = nullptr;
		vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
		vk::raii::SurfaceKHR surface = nullptr;
		vk::raii::PhysicalDevice pDevice = nullptr;
		vk::raii::Device device = nullptr;
		vk::PhysicalDeviceFeatures devFeatures;
		uint32_t graphicsQueueIndex = 0;
		vk::raii::Queue graphicsQueue = nullptr;
		vk::raii::Queue presentQueue = nullptr;

		vk::raii::SwapchainKHR swapchain = nullptr;
		std::vector<vk::Image> swapChainImages;
		vk::SurfaceFormatKHR swapChainSurfaceFormat;
		vk::Extent2D swapChainExtent;
		std::vector<vk::raii::ImageView> swapChainImageViews;

		vk::raii::Buffer vertexBuffer = nullptr;
		vk::raii::DeviceMemory vBufferMemory = nullptr;

		vk::raii::Pipeline graphicsPipeline = nullptr;
		vk::raii::CommandPool commandPool = nullptr;
		std::vector<vk::raii::CommandBuffer> cmdBuffers;
		std::vector<vk::raii::Semaphore> presentCompleteS;
		std::vector<vk::raii::Semaphore> renderFinishedS;
		std::vector<vk::raii::Fence> drawF;
		uint32_t frameIndex = 0;

		bool frameBufferResized = false;

		void initVulkan()
		{
			createInstance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();
			createImageViews();
			createPipeline();
			createCommandPool();
			createVertexBuffer();
			createCommandBuffers();
			createSyncObjects();
		}

		uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags propFlags)
		{
			vk::PhysicalDeviceMemoryProperties memProperties = pDevice.getMemoryProperties();
			for (int i = 0; i < memProperties.memoryTypeCount; ++i) {
				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & propFlags) == propFlags) {
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type!");
		}

		void createVertexBuffer()
		{
			vk::BufferCreateInfo bufferInfo{
				.size = sizeof(vertices[0]) * vertices.size(),
				.usage = vk::BufferUsageFlagBits::eVertexBuffer,
				.sharingMode = vk::SharingMode::eExclusive
			};

			vertexBuffer = vk::raii::Buffer(device, bufferInfo);
			vk::MemoryRequirements memReq= vertexBuffer.getMemoryRequirements();

			vk::MemoryAllocateInfo memInfo{
				.allocationSize = memReq.size,
					.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
			};
			vBufferMemory = vk::raii::DeviceMemory(device, memInfo);
			vertexBuffer.bindMemory(*vBufferMemory, 0);

			void *data = vBufferMemory.mapMemory(0, bufferInfo.size);
			memcpy(data, vertices.data(), bufferInfo.size);
			vBufferMemory.unmapMemory();
		}

		void cleanupSwapchain()
		{
			swapChainImageViews.clear();
			swapchain = nullptr;
		}

		void recreateSwapchain()
		{
			auto [w, h] = window->getFrameBufferSize();
			while(w==0 || h==0)
			{
				auto [tw, th] = window->getFrameBufferSize();
				w=tw; h=th;
				window->pollEvents();
			}

			device.waitIdle();

			cleanupSwapchain();
			createSwapChain();
			createImageViews();
		}

		void drawFrame()
		{
			// wait for previous frame to finish
			// get image from the swapchain
			// record a command buffer
			// submit command buffer
			// present image
			
			// true => wait for all, uint64max = timeout
			auto fenceRes = device.waitForFences(*drawF[frameIndex], vk::True, UINT64_MAX);

			auto [res, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteS[frameIndex], nullptr);
			if(res == vk::Result::eErrorOutOfDateKHR)
			{
				recreateSwapchain();
				return;
			}
			if (res != vk::Result::eSuccess && res != vk::Result::eSuboptimalKHR)
			{
				assert(res == vk::Result::eTimeout || res == vk::Result::eNotReady);
				throw std::runtime_error("Failed to acquire swapchain image!");
			}

			// Make sure to only reset the fence if we are actually rendering
			device.resetFences(*drawF[frameIndex]);

			cmdBuffers[frameIndex].reset();
			recordCommandBuffer(imageIndex);
			
			vk::PipelineStageFlags waitDestStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
			const vk::SubmitInfo submitInfo{
				.waitSemaphoreCount = 1,
					.pWaitSemaphores = &*presentCompleteS[frameIndex],
					.pWaitDstStageMask = &waitDestStageMask, // which stage to wait for the semaphore in
					.commandBufferCount=1,
					.pCommandBuffers=&*cmdBuffers[frameIndex],
					.signalSemaphoreCount=1,
					.pSignalSemaphores=&*renderFinishedS[imageIndex]
			};

			graphicsQueue.submit(submitInfo, *drawF[frameIndex]);

			const vk::PresentInfoKHR presentInfo{
				.waitSemaphoreCount=1,
					.pWaitSemaphores=&*renderFinishedS[imageIndex],
					.swapchainCount=1,
					.pSwapchains=&*swapchain,
					.pImageIndices=&imageIndex
			};

			auto result = presentQueue.presentKHR(presentInfo);
			if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || frameBufferResized)
			{
				frameBufferResized = false;
				recreateSwapchain();
			}
			else if (result != vk::Result::eSuccess)
			{
				throw std::runtime_error("Failed to acquire swapchain image!");
			}
			frameIndex = (frameIndex+1)%MAX_FRAMES_IN_FLIGHT;
		}

		void createSyncObjects()
		{
			assert(presentCompleteS.empty() && renderFinishedS.empty() && drawF.empty());

			for(int i = 0; i < swapChainImages.size(); ++i)
			{
				renderFinishedS.emplace_back(device, vk::SemaphoreCreateInfo());
			}
			for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
			{
				presentCompleteS.emplace_back(device, vk::SemaphoreCreateInfo());
				drawF.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
			}
		}

		void transitionImageLayout(
				uint32_t imageIndex,
				vk::ImageLayout oldLayout,
				vk::ImageLayout newLayout,
				vk::AccessFlags2 srcAccessMask,
				vk::AccessFlags2 dstAccessMask,
				vk::PipelineStageFlags2 srcStageMask,
				vk::PipelineStageFlags2 dstStageMask
				)
		{
			vk::ImageMemoryBarrier2 barrier = {
				.srcStageMask = srcStageMask,
				.srcAccessMask = srcAccessMask,
				.dstStageMask = dstStageMask,
				.dstAccessMask = dstAccessMask,
				.oldLayout = oldLayout,
				.newLayout = newLayout,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = swapChainImages[imageIndex],
				.subresourceRange = {
					.aspectMask = vk::ImageAspectFlagBits::eColor,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				}
			};
			vk::DependencyInfo dependencyInfo = {
				.dependencyFlags = {},
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &barrier
			};
			cmdBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
		}

		void recordCommandBuffer(uint32_t imageIndex)
		{
			cmdBuffers[frameIndex].begin({});
			// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
			transitionImageLayout(
					imageIndex,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eColorAttachmentOptimal,
					{},                                                         // srcAccessMask (no need to wait for previous operations)
					vk::AccessFlagBits2::eColorAttachmentWrite,                 // dstAccessMask
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,         // srcStage
					vk::PipelineStageFlagBits2::eColorAttachmentOutput          // dstStage
			);

			vk::ClearValue clearColor = vk::ClearColorValue(0.005f, 0.005f, 0.005f, 1.0f);
			vk::RenderingAttachmentInfo attachmentInfo = {
				.imageView = swapChainImageViews[imageIndex],
				.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.loadOp = vk::AttachmentLoadOp::eClear, // op before rendering
				.storeOp = vk::AttachmentStoreOp::eStore, // op after rendering
				.clearValue = clearColor
			};

			vk::RenderingInfo renderInfo = {
				.renderArea={.offset={0,0}, .extent=swapChainExtent},
				.layerCount=1,
				.colorAttachmentCount=1,
				.pColorAttachments=&attachmentInfo
			};

			cmdBuffers[frameIndex].beginRendering(renderInfo);
			cmdBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics,*graphicsPipeline);
			
			// viewport + scissor are dynamic so we specify them now
			cmdBuffers[frameIndex].setViewport(0, vk::Viewport(0,0,swapChainExtent.width, swapChainExtent.height, 0, 1));
			cmdBuffers[frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0,0), swapChainExtent));

			cmdBuffers[frameIndex].bindVertexBuffers(0, *vertexBuffer, {0});

			cmdBuffers[frameIndex].draw(vertices.size(), 1, 0, 0);

			cmdBuffers[frameIndex].endRendering();
			
			// After rendering, transition the swapchain image to PRESENT_SRC
			transitionImageLayout(
					imageIndex,
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::ImageLayout::ePresentSrcKHR,
					vk::AccessFlagBits2::eColorAttachmentWrite,             // srcAccessMask
					{},                                                     // dstAccessMask
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,     // srcStage
					vk::PipelineStageFlagBits2::eBottomOfPipe               // dstStage
			);

			cmdBuffers[frameIndex].end();
		}

		void createCommandBuffers()
		{
			vk::CommandBufferAllocateInfo allocInfo{.commandPool=commandPool, .level=vk::CommandBufferLevel::ePrimary,
				.commandBufferCount=MAX_FRAMES_IN_FLIGHT};
			cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);
		}

		void createCommandPool()
		{
			vk::CommandPoolCreateInfo poolInfo{
				.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
					.queueFamilyIndex = graphicsQueueIndex
			};
			commandPool = vk::raii::CommandPool(device, poolInfo);
		}

		void createPipeline()
		{
			auto shaderBin = readFile(SHADER_PATH);
			LOG("Shader size: "<<shaderBin.size()<<"B")
			auto shaderModule = createShaderModule(shaderBin);
			vk::PipelineShaderStageCreateInfo vertInfo{
				.stage = vk::ShaderStageFlagBits::eVertex,
					.module = shaderModule,
					.pName = "vertMain"
			};
			vk::PipelineShaderStageCreateInfo fragInfo{
				.stage = vk::ShaderStageFlagBits::eFragment,
					.module = shaderModule,
					.pName = "fragMain"
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
					.frontFace = vk::FrontFace::eClockwise,
					.depthBiasEnable = vk::False,
					.depthBiasSlopeFactor = 1.0f,
					.lineWidth = 1.0f
			};

			vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples=vk::SampleCountFlagBits::e1,.sampleShadingEnable=vk::False};
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

			vk::raii::PipelineLayout pipelineLayout = nullptr;
			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount=0,.pushConstantRangeCount=0};
			pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

			vk::PipelineRenderingCreateInfo renderingCreateInfo{.colorAttachmentCount=1,.pColorAttachmentFormats=&swapChainSurfaceFormat.format};

			vk::GraphicsPipelineCreateInfo pipelineInfo{
				.pNext = &renderingCreateInfo,
				.stageCount = 2, .pStages = shaderStages,
				.pVertexInputState = &vertInputInfo, .pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState, .pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling, .pColorBlendState = &colorBlending,
				.pDynamicState = &dynamicState, .layout = pipelineLayout, .renderPass = nullptr
			};

			graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &bytes) const
		{
			vk::ShaderModuleCreateInfo info{
				.codeSize = bytes.size() * sizeof(char),
					.pCode = (uint32_t*)bytes.data()
			};
			vk::raii::ShaderModule shaderModule{device, info};
			return shaderModule;
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

		void createImageViews()
		{
			swapChainImageViews.clear();

			vk::ImageViewCreateInfo imViewInfo{
				.viewType = vk::ImageViewType::e2D,
					.format = swapChainSurfaceFormat.format,
					.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0,1,0,1}
			};

			for(auto image : swapChainImages)
			{
				imViewInfo.image = image;
				swapChainImageViews.emplace_back(device, imViewInfo);
			}
		}

		void createSwapChain()
		{
			auto surfCapabilities = pDevice.getSurfaceCapabilitiesKHR(*surface);
			swapChainSurfaceFormat = chooseSwapSurfaceFormat(pDevice.getSurfaceFormatsKHR(*surface));
			swapChainExtent = chooseSwapExtent(surfCapabilities);
			auto minImageCount = std::max(3u, surfCapabilities.minImageCount);
			minImageCount = ( surfCapabilities.maxImageCount > 0 && minImageCount > surfCapabilities.maxImageCount ) ? surfCapabilities.maxImageCount : minImageCount;

			uint32_t imageCount = surfCapabilities.minImageCount + 1;
			// max = 0 means unlimited
			if (surfCapabilities.maxImageCount > 0 && imageCount > surfCapabilities.maxImageCount) {
				imageCount = surfCapabilities.maxImageCount;
			}

			vk::SwapchainCreateInfoKHR swapChainCreateInfo{
				.flags = vk::SwapchainCreateFlagsKHR(),
					.surface = *surface,
					.minImageCount = minImageCount,
					.imageFormat = swapChainSurfaceFormat.format,
					.imageColorSpace = swapChainSurfaceFormat.colorSpace,
					.imageExtent = swapChainExtent,
					.imageArrayLayers =1,
					.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
					.imageSharingMode = vk::SharingMode::eExclusive,
					.preTransform = surfCapabilities.currentTransform,
					.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
					.presentMode = chooseSwapPresentMode(pDevice.getSurfacePresentModesKHR( *surface )),
					.clipped = true,
					.oldSwapchain = nullptr
			};

			swapchain = vk::raii::SwapchainKHR(device,swapChainCreateInfo);
			swapChainImages = swapchain.getImages();
		}

		vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &cap)
		{
			if(cap.currentExtent.width != std::numeric_limits<uint32_t>::max() && cap.currentExtent.height != std::numeric_limits<uint32_t>::max())
				return cap.currentExtent;

			auto [w, h] = window->getFrameBufferSize();
			
			return {
				std::clamp<uint32_t>(w, cap.minImageExtent.width, cap.maxImageExtent.width),
				std::clamp<uint32_t>(h, cap.minImageExtent.height, cap.maxImageExtent.height)
			};
		}

		vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availModes)
		{
			for(auto &mode : availModes)
			{
				if(mode == vk::PresentModeKHR::eMailbox)
					return mode;
			}
			return vk::PresentModeKHR::eFifo;
		}

		vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
		{
			for(const auto &format : availableFormats)
			{
				if(format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
					return format;
			}
			return availableFormats[0];
		}

		void createSurface()
		{
			VkSurfaceKHR _surface;
			if(window->createSurface(*instance, &_surface))
				throw std::runtime_error("Failed to create window surface");

			surface = vk::raii::SurfaceKHR(instance, _surface);
		}

		void createLogicalDevice()
		{
			auto queueFamiliProps = pDevice.getQueueFamilyProperties();
			// get the first index into queueFamilyProperties which supports graphics
			auto graphicsQueueFamilyProperty = std::ranges::find_if( queueFamiliProps, []( auto const & qfp )
					{ return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); } );

			auto graphicsIndex = static_cast<uint32_t>( std::distance( queueFamiliProps.begin(), graphicsQueueFamilyProperty ) );
			graphicsQueueIndex = graphicsIndex;

			// determine a queueFamilyIndex that supports present
			// first check if the graphicsIndex is good enough
			auto presentIndex = pDevice.getSurfaceSupportKHR( graphicsIndex, *surface )
				? graphicsIndex
				: static_cast<uint32_t>( queueFamiliProps.size() );
			if ( presentIndex == queueFamiliProps.size() )
			{
				// the graphicsIndex doesn't support present -> look for another family index that supports both
				// graphics and present
				for ( size_t i = 0; i < queueFamiliProps.size(); i++ )
				{
					if ( ( queueFamiliProps[i].queueFlags & vk::QueueFlagBits::eGraphics ) &&
							pDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
					{
						graphicsIndex = static_cast<uint32_t>( i );
						presentIndex  = graphicsIndex;
						break;
					}
				}
				if ( presentIndex == queueFamiliProps.size() )
				{
					// there's nothing like a single family index that supports both graphics and present -> look for another
					// family index that supports present
					for ( size_t i = 0; i < queueFamiliProps.size(); i++ )
					{
						if ( pDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
						{
							presentIndex = static_cast<uint32_t>( i );
							break;
						}
					}
				}
			}
			if ( ( graphicsIndex == queueFamiliProps.size() ) || ( presentIndex == queueFamiliProps.size() ) )
			{
				throw std::runtime_error( "Could not find a queue for graphics or present -> terminating" );
			}

			float queuePriority = 0.5f;
			vk::DeviceQueueCreateInfo devQueueCreateInfo{
				.queueFamilyIndex=graphicsIndex,
					.queueCount = 1,
					.pQueuePriorities = &queuePriority
			};

			// Create a chain of feature structures
			vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
				{},
				{.shaderDrawParameters=true},
				{.synchronization2=true, .dynamicRendering = true },
				{.extendedDynamicState = true }
			};

			vk::DeviceCreateInfo devCreateInfo{
				.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
					.queueCreateInfoCount=1,
					.pQueueCreateInfos=&devQueueCreateInfo,
					.enabledExtensionCount=static_cast<uint32_t>(deviceExtensions.size()),
					.ppEnabledExtensionNames=deviceExtensions.data()
			};

			device = vk::raii::Device(pDevice, devCreateInfo);
			graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
			presentQueue = vk::raii::Queue( device, presentIndex, 0 );
		}

		uint32_t findQueueFamilies(vk::raii::PhysicalDevice dev)
		{
			// find the index of the first queue family that supports graphics
			std::vector<vk::QueueFamilyProperties> queueFamilyProperties = dev.getQueueFamilyProperties();

			// get the first index into queueFamilyProperties which supports graphics
			auto graphicsQueueFamilyProperty =
				std::find_if( queueFamilyProperties.begin(),
						queueFamilyProperties.end(),
						[]( vk::QueueFamilyProperties const & qfp ) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; } );

			return static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );
		}

		void pickPhysicalDevice()
		{
			auto devices = instance.enumeratePhysicalDevices();

			if (devices.empty()) {
				throw std::runtime_error("failed to find GPUs with Vulkan support!");
			}

			LOG("Available devices:")
			for(auto &dev : devices)
			{
				LOG("\t"<<dev.getProperties().deviceName)
			}

			std::vector<const char*> deviceExtensions = {vk::KHRSwapchainExtensionName};
			const auto devIter = std::ranges::find_if(devices,
				[&](auto const & device) {
					auto queueFamilies = device.getQueueFamilyProperties();
					bool isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;
					const auto qfpIter = std::ranges::find_if(queueFamilies,
							[]( vk::QueueFamilyProperties const & qfp )
							{
								return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
							} );

					isSuitable = isSuitable && ( qfpIter != queueFamilies.end() );
					auto extensions = device.enumerateDeviceExtensionProperties( );
					bool found = true;
					for (auto const & extension : deviceExtensions) {
						auto extensionIter = std::ranges::find_if(extensions, [extension](auto const & ext) {return strcmp(ext.extensionName, extension) == 0;});
						found = found &&  extensionIter != extensions.end();
					}
					isSuitable = isSuitable && found;
					if (isSuitable) {
						pDevice = device;
					}
					return isSuitable;
				});
			if (devIter == devices.end()) {
				throw std::runtime_error("failed to find a suitable GPU!");
			}
		}

		void createInstance()
		{
			constexpr vk::ApplicationInfo appInfo{
				.pApplicationName = "Triangle",
					.applicationVersion = VK_MAKE_VERSION(1,0,0),
					.pEngineName = "No Engine",
					.engineVersion = VK_MAKE_VERSION(1,0,0),
					.apiVersion = vk::ApiVersion14
			};

			std::vector<char const*> requiredLayers;
			if(_ENABLE_VALIDATION_LAYERS)
			{
				LOG("Adding validation layers...")
				requiredLayers.assign(validationLayers.begin(), validationLayers.end());
			}

			auto layerProperties = context.enumerateInstanceLayerProperties();
			if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
						return std::ranges::none_of(layerProperties,
								[requiredLayer](auto const& layerProperty)
								{ return strcmp(layerProperty.layerName, requiredLayer) == 0; });
						}))
			{
				throw std::runtime_error("One or more required layers are not supported!");
			}
			
			auto requiredExtensions = getRequiredExtensions();

			// Check if the required extensions are supported by the Vulkan implementation.
			auto extensionProperties = context.enumerateInstanceExtensionProperties();
			for (auto &extension : requiredExtensions)
			{
				if (std::ranges::none_of(extensionProperties,
							[extension](auto const& extensionProperty)
							{ return strcmp(extensionProperty.extensionName, extension) == 0; }))
				{
					throw std::runtime_error("Required GLFW extension not supported: " + std::string(extension));
				}
			}

			vk::InstanceCreateInfo createInfo{
				.pApplicationInfo=&appInfo,
					.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
					.ppEnabledLayerNames = requiredLayers.data(),
					.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
					.ppEnabledExtensionNames = requiredExtensions.data()
			};

			instance = vk::raii::Instance(context, createInfo);
		}

		static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
			LOG("Validation layer: type " << to_string(type) << " Message: " << pCallbackData->pMessage)

			return vk::False;
		}

		void setupDebugMessenger()
		{
			if(!_ENABLE_VALIDATION_LAYERS) return;
			LOG("Setting up the debug messenger")
			vk::DebugUtilsMessageSeverityFlagsEXT sflags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose|vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning|vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
			vk::DebugUtilsMessageTypeFlagsEXT tflags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
			vk::DebugUtilsMessengerCreateInfoEXT msgCreateInfo{
				.messageSeverity = sflags,
					.messageType = tflags,
					.pfnUserCallback = &debugCallback
			};
			debugMessenger = instance.createDebugUtilsMessengerEXT(msgCreateInfo);
		}

		std::vector<const char*> getRequiredExtensions()
		{
			uint32_t extCount = 0;
			const char * const*ext;

			ext = Window::getInstanceExtensions(&extCount);

			std::vector extensions(ext, ext + extCount);
			if (_ENABLE_VALIDATION_LAYERS) {
				extensions.push_back(vk::EXTDebugUtilsExtensionName );
			}

			return extensions;
		}

		void mainLoop()
		{
			while(window->isRunning())
			{
				window->pollEvents();
				drawFrame();
			}
			device.waitIdle();

		}

		void initWindow()
		{
			window = Window::create({
				.name="Vulkan Testing",
				.width=WIDTH,
				.height=HEIGHT,
				.frameBufferResizeCallback=&frameBufferResized}
			);

		}

		void cleanup()
		{
			// After cleaning up the swapchain glfwTerminate and SDL_Quit 
			// no longer cause a segmentation fault :))
			cleanupSwapchain();
			delete window;
		}
};

int main()
{
	
	App app;
	try {
		app.run();
	} catch (const std::exception &e) {
		std::cout<<e.what()<<std::endl;
		return 1;
	}
}
