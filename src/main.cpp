#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Window.h"

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>

#define LOG(x) std::cout<<x<<std::endl;

#ifdef _DEBUG
constexpr bool _ENABLE_VALIDATION_LAYERS = true;
#else
constexpr bool _ENABLE_VALIDATION_LAYERS = false;
#endif

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

constexpr const char* SHADER_PATH = "shaders/texture.spv";
constexpr const char* TEXTURE_PATH = "textures/texture2.png";

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDesc()
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
	}
	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescs()
	{
		return {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
				vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
				vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
		};
	}
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const std::vector<Vertex> triangle = {
	{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

const std::vector<Vertex> rect = {
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}}, 
	{{0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
};
const std::vector<uint16_t> rectIndices = {
	0, 1, 2, 2, 3, 0
};

const std::vector<Vertex> fullSquare = {
	{{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
	{{1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
	{{-1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
	{{-1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
	{{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
	{{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}}
};

auto vertices = rect;
auto indices = rectIndices;

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
		vk::raii::Buffer indexBuffer = nullptr;
		vk::raii::DeviceMemory iBufferMemory = nullptr;

		std::vector<vk::raii::Buffer> uniformBuffers;
		std::vector<vk::raii::DeviceMemory> uBuffersMemory;
		std::vector<void*> uBuffersMapped;

		vk::raii::DescriptorSetLayout descSetLayout = nullptr;
		vk::raii::PipelineLayout pipelineLayout = nullptr;
		vk::raii::Pipeline graphicsPipeline = nullptr;

		vk::raii::DescriptorPool descPool = nullptr;
		std::vector<vk::raii::DescriptorSet> descSets;

		vk::raii::CommandPool commandPool = nullptr;
		std::vector<vk::raii::CommandBuffer> cmdBuffers;
		std::vector<vk::raii::Semaphore> presentCompleteS;
		std::vector<vk::raii::Semaphore> renderFinishedS;
		std::vector<vk::raii::Fence> drawF;
		uint32_t frameIndex = 0;

		vk::raii::Image textureImage = nullptr;
		vk::raii::ImageView textureImageView = nullptr;
		vk::raii::DeviceMemory textureMemory = nullptr;
		vk::raii::Sampler textureSampler = nullptr;

		bool frameBufferResized = false;

		std::unique_ptr<Window> window;

		void initVulkan()
		{
			createInstance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();
			createImageViews();
			createDescSetLayout();
			createPipeline();
			createCommandPool();
			createTextureImage();
			createTextureImageView();
			createTextureSampler();
			createVertexBuffer();
			createIndexBuffer();
			createUniformBuffers();
			createDescPool();
			createDescSets();
			createCommandBuffers();
			createSyncObjects();
		}

		void createTextureSampler()
		{
			vk::PhysicalDeviceProperties properties = pDevice.getProperties();
			vk::SamplerCreateInfo        samplerInfo{
				.magFilter        = vk::Filter::eLinear,
					.minFilter        = vk::Filter::eLinear,
					.mipmapMode       = vk::SamplerMipmapMode::eLinear,
					.addressModeU     = vk::SamplerAddressMode::eRepeat,
					.addressModeV     = vk::SamplerAddressMode::eRepeat,
					.addressModeW     = vk::SamplerAddressMode::eRepeat,
					.mipLodBias       = 0.0f,
					.anisotropyEnable = vk::True,
					.maxAnisotropy    = properties.limits.maxSamplerAnisotropy,
					.compareEnable    = vk::False,
					.compareOp        = vk::CompareOp::eAlways,
					.unnormalizedCoordinates = vk::False
			};

			textureSampler = vk::raii::Sampler(device, samplerInfo);
		}

		void createTextureImageView()
		{
			textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
		}

		vk::raii::ImageView createImageView(vk::raii::Image &image, vk::Format format)
		{
			vk::ImageViewCreateInfo viewInfo{ .image = image, .viewType = vk::ImageViewType::e2D,
				.format = format, .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
			return std::move(vk::raii::ImageView( device, viewInfo ));
		}

		void copyBufferToImage(const vk::raii::Buffer &buf, vk::raii::Image &image, uint32_t w, uint32_t h)
		{
			auto cmdBuffer = beginSingleTimeCommands();

			vk::BufferImageCopy region{.bufferOffset=0, .bufferRowLength=0, .bufferImageHeight=0,
				.imageSubresource={vk::ImageAspectFlagBits::eColor, 0, 0, 1},
				.imageOffset={0,0,0},
				.imageExtent={w,h,1}
			};
			cmdBuffer.copyBufferToImage(buf, image, vk::ImageLayout::eTransferDstOptimal, {region});

			endSingleTimeCommands(cmdBuffer);
		}

		void transitionLayout(const vk::raii::Image &image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
		{
			auto cmdBuffer = beginSingleTimeCommands();
			vk::ImageMemoryBarrier barrier{
				.oldLayout=oldLayout,
				.newLayout = newLayout,
				.image=image,
				.subresourceRange={vk::ImageAspectFlagBits::eColor, 0,1,0,1}
			};

			vk::PipelineStageFlags sourceStage;
			vk::PipelineStageFlags destinationStage;

			if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
				barrier.srcAccessMask = {};
				barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

				sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
				destinationStage = vk::PipelineStageFlagBits::eTransfer;
			} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
				barrier.srcAccessMask =  vk::AccessFlagBits::eTransferWrite;
				barrier.dstAccessMask =  vk::AccessFlagBits::eShaderRead;

				sourceStage = vk::PipelineStageFlagBits::eTransfer;
				destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
			} else {
				throw std::invalid_argument("unsupported layout transition!");
			}

			cmdBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);

			endSingleTimeCommands(cmdBuffer);
		}

		vk::raii::CommandBuffer beginSingleTimeCommands()
		{
			vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
			vk::raii::CommandBuffer cmdBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());

			cmdBuffer.begin(vk::CommandBufferBeginInfo { .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

			return std::move(cmdBuffer);
		}

		void endSingleTimeCommands(vk::raii::CommandBuffer &cmdBuffer)
		{
			cmdBuffer.end();
			graphicsQueue.submit(vk::SubmitInfo{
					.commandBufferCount=1,
					.pCommandBuffers=&*cmdBuffer},
					nullptr
				);

			graphicsQueue.waitIdle();
		}

		void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
			vk::ImageCreateInfo imageInfo{ .imageType = vk::ImageType::e2D, .format = format,
				.extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
				.samples = vk::SampleCountFlagBits::e1, .tiling = tiling,
				.usage = usage, .sharingMode = vk::SharingMode::eExclusive };

			image = vk::raii::Image(device, imageInfo);

			vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
			vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size,
				.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
			imageMemory = vk::raii::DeviceMemory(device, allocInfo);
			image.bindMemory(imageMemory, 0);
		}

		void createTextureImage()
		{
			int tWidth, tHeight, tChannels;
			stbi_uc *pixels = stbi_load(TEXTURE_PATH, &tWidth, &tHeight, &tChannels, STBI_rgb_alpha);
			vk::DeviceSize imageSize = tWidth*tHeight * 4;

			if(!pixels) throw std::runtime_error("Failed to load texture image!");

			vk::raii::Buffer stagingBuffer({});
			vk::raii::DeviceMemory memory({});

			createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible |
					vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, memory);

			void* data = memory.mapMemory(0, imageSize);
			memcpy(data, pixels, imageSize);
			memory.unmapMemory();

			stbi_image_free(pixels);

			createImage(tWidth, tHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst|
					vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureMemory);

			transitionLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
			copyBufferToImage(stagingBuffer, textureImage, tWidth, tHeight);
			transitionLayout(textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		}

		void createDescSets()
		{
			std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descSetLayout);
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = descPool,
					.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
					.pSetLayouts = layouts.data()
			};

			descSets.clear();
			descSets = device.allocateDescriptorSets(allocInfo);

			for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
			{
				vk::DescriptorBufferInfo bufferInfo{ .buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject) };
				vk::DescriptorImageInfo imageInfo{.sampler=textureSampler, .imageView=textureImageView, .imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal};

				std::array descriptorWrites = {
					vk::WriteDescriptorSet{ .dstSet = descSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo },
					vk::WriteDescriptorSet{ .dstSet = descSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &imageInfo }
				};
				device.updateDescriptorSets(descriptorWrites, {});
			}
		}

		void createDescPool()
		{
			std::array poolSize = {
				vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
				vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)
			};

			vk::DescriptorPoolCreateInfo poolInfo{
				.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
					.maxSets=MAX_FRAMES_IN_FLIGHT,
					.poolSizeCount=poolSize.size(),
					.pPoolSizes=poolSize.data(),
			};
			
			descPool = vk::raii::DescriptorPool(device, poolInfo);
		}

		void updateUniformBuffer(uint32_t currentImage)
		{
			static auto startTime = std::chrono::high_resolution_clock::now();

			auto currentTime = std::chrono::high_resolution_clock::now();
			float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
			
			UniformBufferObject ubo{};
			ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
			ubo.proj[1][1] *= -1; // otherwise it would be upside down

			memcpy(uBuffersMapped[currentImage], &ubo, sizeof(ubo));
		}

		void createUniformBuffers()
		{
			uniformBuffers.clear();
			uBuffersMemory.clear();
			uBuffersMapped.clear();

			for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
			{
				vk::DeviceSize bufSize = sizeof(UniformBufferObject);
				vk::raii::Buffer buffer({});
				vk::raii::DeviceMemory memory({});

				createBuffer(bufSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible|
						vk::MemoryPropertyFlagBits::eHostCoherent, buffer, memory);

				uniformBuffers.emplace_back(std::move(buffer));
				uBuffersMemory.emplace_back(std::move(memory));
				uBuffersMapped.emplace_back(uBuffersMemory[i].mapMemory(0, bufSize));
			}
		}

		void createDescSetLayout()
		{
			std::array bindings = {
				vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
				vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
			};

			vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount=bindings.size(), .pBindings=bindings.data()};
			descSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
		}

		void createIndexBuffer()
		{
			vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

			vk::raii::Buffer stagingBuffer({});
			vk::raii::DeviceMemory stagingBufferMemory({});
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

			void* data = stagingBufferMemory.mapMemory(0, bufferSize);
			memcpy(data, indices.data(), (size_t) bufferSize);
			stagingBufferMemory.unmapMemory();

			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, iBufferMemory);

			copyBuffer(stagingBuffer, indexBuffer, bufferSize);
		}

		void copyBuffer(vk::raii::Buffer &srcBuf, vk::raii::Buffer &dstBuf, vk::DeviceSize size)
		{
			auto commandCopyBuffer = beginSingleTimeCommands();
			commandCopyBuffer.copyBuffer(srcBuf, dstBuf, vk::BufferCopy(0, 0, size));
			endSingleTimeCommands(commandCopyBuffer);
		}

		void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMem)
		{
			vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
			buffer = vk::raii::Buffer(device, bufferInfo);
			vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
			vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
			bufferMem= vk::raii::DeviceMemory(device, allocInfo);
			buffer.bindMemory(*bufferMem, 0);
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
			vk::DeviceSize bufSize = sizeof(vertices[0]) * vertices.size();
			vk::raii::Buffer       stagingBuffer({});
			vk::raii::DeviceMemory stagingBufferMemory({});
			createBuffer(bufSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

			void *dataStaging = stagingBufferMemory.mapMemory(0, bufSize);
			memcpy(dataStaging, vertices.data(), bufSize);
			stagingBufferMemory.unmapMemory();

			createBuffer(bufSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal,
					vertexBuffer, vBufferMemory);

			copyBuffer(stagingBuffer, vertexBuffer, bufSize);
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

			updateUniformBuffer(frameIndex);

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
			cmdBuffers[frameIndex].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);

			cmdBuffers[frameIndex].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descSets[frameIndex], nullptr);
			cmdBuffers[frameIndex].drawIndexed(indices.size(), 1, 0, 0, 0);

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
					.frontFace = vk::FrontFace::eCounterClockwise,
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

			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount=1,
				.pSetLayouts=&*descSetLayout,			
				.pushConstantRangeCount=0
			};
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
			auto queueFamilyProps = pDevice.getQueueFamilyProperties();
			// get the first index into queueFamilyProperties which supports graphics
			auto graphicsQueueFamilyProperty = std::ranges::find_if( queueFamilyProps, []( auto const & qfp )
					{ return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); } );

			auto graphicsIndex = static_cast<uint32_t>( std::distance( queueFamilyProps.begin(), graphicsQueueFamilyProperty ) );
			graphicsQueueIndex = graphicsIndex;

			// determine a queueFamilyIndex that supports present
			// first check if the graphicsIndex is good enough
			auto presentIndex = pDevice.getSurfaceSupportKHR( graphicsIndex, *surface )
				? graphicsIndex
				: static_cast<uint32_t>( queueFamilyProps.size() );
			if ( presentIndex == queueFamilyProps.size() )
			{
				// the graphicsIndex doesn't support present -> look for another family index that supports both
				// graphics and present
				for ( size_t i = 0; i < queueFamilyProps.size(); i++ )
				{
					if ( ( queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics ) &&
							pDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
					{
						graphicsIndex = static_cast<uint32_t>( i );
						presentIndex  = graphicsIndex;
						break;
					}
				}
				if ( presentIndex == queueFamilyProps.size() )
				{
					// there's nothing like a single family index that supports both graphics and present -> look for another
					// family index that supports present
					for ( size_t i = 0; i < queueFamilyProps.size(); i++ )
					{
						if ( pDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
						{
							presentIndex = static_cast<uint32_t>( i );
							break;
						}
					}
				}
			}
			if ( ( graphicsIndex == queueFamilyProps.size() ) || ( presentIndex == queueFamilyProps.size() ) )
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
				{.features = {.samplerAnisotropy = true}},
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
			window = std::unique_ptr<Window>(Window::create({
				.name="Vulkan Testing",
				.width=WIDTH,
				.height=HEIGHT,
				.frameBufferResizeCallback=&frameBufferResized}
			));

		}

		void cleanup()
		{
			cleanupSwapchain();
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
