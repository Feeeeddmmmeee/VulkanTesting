#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Window.h"
#include "Pipeline.h"
#include "Vertex.h"
#include "Models.h"
#include "Camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // convert from opengl's -1 - 1 depth to vulkan's 0 - 1
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>

#define LOG(x) std::cout<<x<<std::endl;
#include <SDL3/SDL.h>

#ifdef _DEBUG
constexpr bool _ENABLE_VALIDATION_LAYERS = true;
#else
constexpr bool _ENABLE_VALIDATION_LAYERS = false;
#endif

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
constexpr uint32_t MAX_OBJECTS = 4;

constexpr const char* TEXTURE_PATH = "textures/viking_room.png";
constexpr bool ENABLE_MSAA = true;

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName
};

struct Object 
{
	glm::vec3 pos = {0,0,0};
	glm::vec3 rotation = {0,0,0};
	glm::vec3 scale = {1,1,1};

	Model model;

	// one for each frame in flight
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uBuffersMemory;
	std::vector<void*> uBuffersMapped;

	std::vector<vk::raii::DescriptorSet> descriptorSets;

	glm::mat4 getModelMatrix() const 
	{
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, pos);
        model = glm::rotate(model, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, scale);
        return model;
    }
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

		vk::raii::DescriptorSetLayout descSetLayout = nullptr;

		vk::raii::DescriptorPool descPool = nullptr;
		std::array<Object, MAX_OBJECTS> objects;

		vk::raii::CommandPool commandPool = nullptr;
		std::vector<vk::raii::CommandBuffer> cmdBuffers;
		std::vector<vk::raii::Semaphore> presentCompleteS;
		std::vector<vk::raii::Semaphore> renderFinishedS;
		std::vector<vk::raii::Fence> drawF;
		uint32_t frameIndex = 0;

		uint32_t mipLevels;
		vk::raii::Image textureImage = nullptr;
		vk::raii::ImageView textureImageView = nullptr;
		vk::raii::DeviceMemory textureMemory = nullptr;
		vk::raii::Sampler textureSampler = nullptr;

		vk::raii::Image depthImage = nullptr;
		vk::raii::DeviceMemory depthImageMemory = nullptr;
		vk::raii::ImageView depthImageView = nullptr;

		bool frameBufferResized = false;

		std::unique_ptr<Window> window;
		std::unique_ptr<Camera> camera;
		std::unique_ptr<PipelineManager> pipelineManager;

		vk::raii::Image colorImage = nullptr;
		vk::raii::DeviceMemory colorImageMemory = nullptr;
		vk::raii::ImageView colorImageView = nullptr;
		vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

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
			setupPipelineManager();
			createCommandPool();
			createColorResources();
			createDepthResources();
			createTextureImage();
			createTextureImageView();
			createTextureSampler();
			setupCamera();
			setupObjects();
			createUniformBuffers();
			createDescPool();
			createDescSets();
			createCommandBuffers();
			createSyncObjects();
		}

		vk::SampleCountFlagBits getMaxUsableSampleCount()
		{
			if(!ENABLE_MSAA) return vk::SampleCountFlagBits::e1;

			vk::PhysicalDeviceProperties props = pDevice.getProperties();
			vk::SampleCountFlags counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;

			if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
			if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
			if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
			if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
			if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
			if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

			return vk::SampleCountFlagBits::e1;
		}

		void createColorResources()
		{
			vk::Format format = swapChainSurfaceFormat.format;

			createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, format, vk::ImageTiling::eOptimal,
					vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,  vk::MemoryPropertyFlagBits::eDeviceLocal,
					colorImage, colorImageMemory
					);
			colorImageView = createImageView(colorImage, format, vk::ImageAspectFlagBits::eColor, 1);
		}

		void generateMipMaps(vk::raii::Image &image, vk::Format format, uint32_t w, uint32_t h, uint32_t mipLevels)
		{
			vk::FormatProperties formatProperties = pDevice.getFormatProperties(format);
			if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
				throw std::runtime_error("Texture image format does not support linear blitting!");

			auto cmdBuffer = beginSingleTimeCommands();
			vk::ImageMemoryBarrier barrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
				.dstAccessMask=vk::AccessFlagBits::eTransferRead, .oldLayout=vk::ImageLayout::eTransferDstOptimal,
				.newLayout=vk::ImageLayout::eTransferSrcOptimal, .srcQueueFamilyIndex=vk::QueueFamilyIgnored,
				.dstQueueFamilyIndex=vk::QueueFamilyIgnored, .image=image
			};
			barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.subresourceRange.levelCount = 1;

			auto mipW = w, mipH = h;
			for(uint32_t i = 1; i < mipLevels; ++i)
			{
				barrier.subresourceRange.baseMipLevel = i - 1;
				barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
				barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
				barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

				cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

				vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
				offsets[0] = vk::Offset3D(0, 0, 0);
				offsets[1] = vk::Offset3D(mipW, mipH, 1);
				dstOffsets[0] = vk::Offset3D(0, 0, 0);
				dstOffsets[1] = vk::Offset3D(mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1, 1);
				vk::ImageBlit blit = { .srcSubresource = {}, .srcOffsets = offsets,
					.dstSubresource =  {}, .dstOffsets = dstOffsets };
				blit.srcSubresource = vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
				blit.dstSubresource = vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, i, 0, 1);

				cmdBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, { blit }, vk::Filter::eLinear);

				barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
				barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
				barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
				barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

				cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);
				if (mipW > 1) mipW /= 2;
				if (mipH > 1) mipH /= 2;
			}

			barrier.subresourceRange.baseMipLevel = mipLevels - 1;
			barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

			endSingleTimeCommands(cmdBuffer);
		}

		void setupPipelineManager()
		{
			this->pipelineManager = std::make_unique<PipelineManager>(
					device,
					descSetLayout,
					findDepthFormat(),
					swapChainSurfaceFormat,
					msaaSamples
			);
		}

		void updateCamera()
		{
			camera->height = swapChainExtent.height;
			camera->width = swapChainExtent.width;
		}

		void setupCamera()
		{
			camera = std::make_unique<Camera>(swapChainExtent.width, swapChainExtent.height, 45.0f, glm::vec3{-2.6f, 0.2f, 0.3f}, glm::vec3{0.9f, -0.1f, 0.2f});
		}

		void createMeshIndexBuffer(Mesh &mesh, std::vector<uint32_t> &indices)
		{
			vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

			vk::raii::Buffer stagingBuffer({});
			vk::raii::DeviceMemory stagingBufferMemory({});
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible |
					vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

			void* data = stagingBufferMemory.mapMemory(0, bufferSize);
			memcpy(data, indices.data(), (size_t) bufferSize);
			stagingBufferMemory.unmapMemory();

			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
					vk::MemoryPropertyFlagBits::eDeviceLocal, mesh.indexBuffer, mesh.iBufferMemory);

			copyBuffer(stagingBuffer, mesh.indexBuffer, bufferSize);
		}

		void createMeshVertexBuffer(Mesh &mesh, std::vector<Vertex> &vertices)
		{
			vk::DeviceSize bufSize = sizeof(vertices[0]) * vertices.size();

			vk::raii::Buffer       stagingBuffer({});
			vk::raii::DeviceMemory stagingBufferMemory({});
			createBuffer(bufSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible |
					vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

			void *dataStaging = stagingBufferMemory.mapMemory(0, bufSize);
			memcpy(dataStaging, vertices.data(), bufSize);
			stagingBufferMemory.unmapMemory();

			createBuffer(bufSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
					vk::MemoryPropertyFlagBits::eDeviceLocal, mesh.vertexBuffer, mesh.vBufferMemory);

			copyBuffer(stagingBuffer, mesh.vertexBuffer, bufSize);
		}

		void setupObjects()
		{
			objects[0].pos= {0, -5, 0};
			objects[0].scale = {.5,.5,.5};
			loadModel(objects[0].model, "models/viking_room.obj");
			objects[0].model.meshes[0].material.setPipeline(pipelineManager->get({
						.vertMain="vertMain",
						.fragMain="fragMain",
						.vert="shaders/texture.spv",
						.frag="shaders/texture.spv"
					}));
			
			loadModel(objects[1].model, "models/sponza.obj", true, true);
			objects[1].scale = {0.2,0.2,0.2};
			objects[1].model.meshes[0].material.setPipeline(pipelineManager->get({
						.vertMain="vertMain",
						.fragMain="fragMain",
						.vert="shaders/uv.spv",
						.frag="shaders/uv.spv"
					}));

			objects[2].pos = {0,5,0};
			objects[2].scale = {.007,.007,.007};
			loadModel(objects[2].model, "models/teapot.obj", true, true);
			objects[2].model.meshes[0].material.setPipeline(pipelineManager->get({
						.vertMain="vertMain",
						.fragMain="fragMain",
						.vert="shaders/uv.spv",
						.frag="shaders/uv.spv"
					}));

			objects[3].pos = {0,7,0};
			loadModel(objects[3].model, "models/dragon.obj", true, true);
			objects[3].scale = {1.5,1.5,1.5};
			objects[3].model.meshes[0].material.setPipeline(pipelineManager->get({
						.vertMain="vertMain",
						.fragMain="fragMain",
						.vert="shaders/projected.spv",
						.frag="shaders/projected.spv"
					}));
		}

		void loadModel(Model &model, const char *path, bool swapYZ = false, bool flipTriangles = false)
		{
			LOG("Loading model: "<<path<<"...")
			tinyobj::attrib_t attrib;
			std::vector<tinyobj::shape_t> shapes;
			std::vector<tinyobj::material_t> materials;
			std::string warn, err;
			std::vector<Vertex> vertices;
			std::vector<uint32_t> indices;

			if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path)) {
				throw std::runtime_error(warn + err);
			}

			std::unordered_map<Vertex, uint32_t> uniqueVerts{};

			for (const auto& shape : shapes) {
				for (const auto& index : shape.mesh.indices) {
					Vertex vertex{};

					if(swapYZ)
					{
						vertex.pos = {
							attrib.vertices[3 * index.vertex_index + 0],
							attrib.vertices[3 * index.vertex_index + 2],
							attrib.vertices[3 * index.vertex_index + 1]
						};
					}
					else
					{
						vertex.pos = {
							attrib.vertices[3 * index.vertex_index + 0],
							attrib.vertices[3 * index.vertex_index + 1],
							attrib.vertices[3 * index.vertex_index + 2]
						};
					}

					if(attrib.texcoords.size())
					{
						vertex.texCoord = {
							attrib.texcoords[2 * index.texcoord_index + 0],
							1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
						};
					}

					vertex.color = {1.0f, 1.0f, 1.0f};
					if(!uniqueVerts.contains(vertex))
					{
						uniqueVerts[vertex] = vertices.size();
						vertices.push_back(vertex);
					}
					indices.push_back(uniqueVerts[vertex]);
				}
			}

			if(flipTriangles)
			{
				for(int i = 0; i < indices.size(); i+=3)
					std::swap(indices[i+1], indices[i+2]);
			}

			model.meshes.push_back(Mesh{.vertexCount=(uint32_t)indices.size()});
			createMeshVertexBuffer(model.meshes[0], vertices);
			createMeshIndexBuffer(model.meshes[0], indices);
		}

		bool hasStencilComponent(vk::Format format) {
			return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
		}

		vk::Format findDepthFormat()
		{
			return findSupportedFormat(
					{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
					vk::ImageTiling::eOptimal,
					vk::FormatFeatureFlagBits::eDepthStencilAttachment
				);
		}

		vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
		{
			for(auto format : candidates)
			{
				vk::FormatProperties props = pDevice.getFormatProperties(format);
				if ((tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) ||
					(tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)) {
					return format;
				}
			}

			throw std::runtime_error("Failed to find supported format!");
		}

		void createDepthResources()
		{
			auto depthFormat = findDepthFormat();
			createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, 
					vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
					vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory
				);
			depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
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
					.minLod = 0,
					.maxLod = vk::LodClampNone,
					.unnormalizedCoordinates = vk::False
			};

			textureSampler = vk::raii::Sampler(device, samplerInfo);
		}

		void createTextureImageView()
		{
			textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
		}

		vk::raii::ImageView createImageView(vk::raii::Image &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
		{
			vk::ImageViewCreateInfo viewInfo{ .image = image, .viewType = vk::ImageViewType::e2D,
				.format = format, .subresourceRange = { aspectFlags, 0, mipLevels, 0, 1 } };
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

		void transitionLayout(const vk::raii::Image &image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
		{
			auto cmdBuffer = beginSingleTimeCommands();
			vk::ImageMemoryBarrier barrier{
				.oldLayout=oldLayout,
				.newLayout = newLayout,
				.image=image,
				.subresourceRange={vk::ImageAspectFlagBits::eColor, 0,mipLevels,0,1}
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

		void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits sampleCount, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
			vk::ImageCreateInfo imageInfo{ .imageType = vk::ImageType::e2D, .format = format,
				.extent = {width, height, 1}, .mipLevels = mipLevels, .arrayLayers = 1,
				.samples = sampleCount, .tiling = tiling,
				.usage = usage, .sharingMode = vk::SharingMode::eExclusive
			};

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
			mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(tWidth, tHeight)))) + 1;
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

			createImage(tWidth, tHeight, mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst|
					vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureMemory);

			transitionLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
			copyBufferToImage(stagingBuffer, textureImage, tWidth, tHeight);
			generateMipMaps(textureImage, vk::Format::eR8G8B8A8Srgb, tWidth, tHeight, mipLevels);
		}

		void createDescSets()
		{
			for(auto &object : objects)
			{
				std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descSetLayout);
				vk::DescriptorSetAllocateInfo allocInfo{
					.descriptorPool = descPool,
						.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
						.pSetLayouts = layouts.data()
				};

				object.descriptorSets.clear();
				object.descriptorSets = device.allocateDescriptorSets(allocInfo);

				for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
				{
					vk::DescriptorBufferInfo bufferInfo{ .buffer = object.uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject) };
					vk::DescriptorImageInfo imageInfo{.sampler=textureSampler, .imageView=textureImageView, .imageLayout=vk::ImageLayout::eShaderReadOnlyOptimal};

					std::array descriptorWrites = {
						vk::WriteDescriptorSet{ .dstSet = object.descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
							.descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo },
						vk::WriteDescriptorSet{ .dstSet = object.descriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
							.descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &imageInfo }
					};
					device.updateDescriptorSets(descriptorWrites, {});
				}
			}
		}

		void createDescPool()
		{
			std::array poolSize = {
				vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT*MAX_OBJECTS),
				vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT*MAX_OBJECTS)
			};

			vk::DescriptorPoolCreateInfo poolInfo{
				.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
					.maxSets=MAX_FRAMES_IN_FLIGHT*MAX_OBJECTS,
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
			
			for(auto &object : objects)
			{
				// object.rotation.z += 0.00005f;
				UniformBufferObject ubo {
					.model = object.getModelMatrix(),
					.view = camera->getViewMatrix(),
						.proj = camera->getProjMatrix()
				};
				memcpy(object.uBuffersMapped[currentImage], &ubo, sizeof(ubo));
			}
		}

		void createUniformBuffers()
		{
			for(auto &object : objects)
			{
				object.uniformBuffers.clear();
				object.uBuffersMemory.clear();
				object.uBuffersMapped.clear();

				for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
				{
					vk::DeviceSize bufSize = sizeof(UniformBufferObject);
					vk::raii::Buffer buffer({});
					vk::raii::DeviceMemory memory({});

					createBuffer(bufSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible|
							vk::MemoryPropertyFlagBits::eHostCoherent, buffer, memory);

					object.uniformBuffers.emplace_back(std::move(buffer));
					object.uBuffersMemory.emplace_back(std::move(memory));
					object.uBuffersMapped.emplace_back(object.uBuffersMemory[i].mapMemory(0, bufSize));
				}
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
			createColorResources();
			createDepthResources();
			updateCamera();
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
				vk::Image image,
				vk::ImageLayout oldLayout,
				vk::ImageLayout newLayout,
				vk::AccessFlags2 srcAccessMask,
				vk::AccessFlags2 dstAccessMask,
				vk::PipelineStageFlags2 srcStageMask,
				vk::PipelineStageFlags2 dstStageMask,
				vk::ImageAspectFlags aspectFlags
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
				.image = image,
				.subresourceRange = {
					.aspectMask = aspectFlags,
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
					swapChainImages[imageIndex],
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eColorAttachmentOptimal,
					{},                                                         // srcAccessMask (no need to wait for previous operations)
					vk::AccessFlagBits2::eColorAttachmentWrite,                 // dstAccessMask
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,         // srcStage
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,          // dstStage
					vk::ImageAspectFlagBits::eColor
			);
			transitionImageLayout(
					*colorImage,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eColorAttachmentOptimal,
					{},                                                         // srcAccessMask (no need to wait for previous operations)
					vk::AccessFlagBits2::eColorAttachmentWrite,                 // dstAccessMask
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,         // srcStage
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,          // dstStage
					vk::ImageAspectFlagBits::eColor
			);
			// New transition for the depth image
			transitionImageLayout(
					*depthImage,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eDepthAttachmentOptimal,
					vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
					vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
					vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
					vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
					vk::ImageAspectFlagBits::eDepth);

			vk::ClearValue clearColor = vk::ClearColorValue(0.005f, 0.005f, 0.005f, 1.0f);
			vk::RenderingAttachmentInfo attachmentInfo = {
				.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
				.loadOp = vk::AttachmentLoadOp::eClear, // op before rendering
				.storeOp = vk::AttachmentStoreOp::eStore, // op after rendering
				.clearValue = clearColor
			};

			if (msaaSamples == vk::SampleCountFlagBits::e1) {
				attachmentInfo.imageView = swapChainImageViews[imageIndex];
				attachmentInfo.resolveMode = vk::ResolveModeFlagBits::eNone;
				attachmentInfo.resolveImageView = VK_NULL_HANDLE;
			} 
			else 
			{
				attachmentInfo.imageView = colorImageView;
				attachmentInfo.resolveMode = vk::ResolveModeFlagBits::eAverage;
				attachmentInfo.resolveImageView = swapChainImageViews[imageIndex];
				attachmentInfo.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
			}

			vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
			vk::RenderingAttachmentInfo depthInfo = {
				.imageView = depthImageView,
				.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
				.loadOp = vk::AttachmentLoadOp::eClear,
				.storeOp = vk::AttachmentStoreOp::eDontCare,
				.clearValue = clearDepth
			};

			vk::RenderingInfo renderInfo = {
				.renderArea={.offset={0,0}, .extent=swapChainExtent},
				.layerCount=1,
				.colorAttachmentCount=1,
				.pColorAttachments=&attachmentInfo,
				.pDepthAttachment=&depthInfo
			};

			cmdBuffers[frameIndex].beginRendering(renderInfo);
			
			// viewport + scissor are dynamic so we specify them now
			cmdBuffers[frameIndex].setViewport(0, vk::Viewport(0,0,swapChainExtent.width, swapChainExtent.height, 0, 1));
			cmdBuffers[frameIndex].setScissor(0, vk::Rect2D(vk::Offset2D(0,0), swapChainExtent));

			for(auto &object : objects)
			{
				for(auto &mesh : object.model.meshes)
				{
					auto pipeline = mesh.material.pipeline;
					cmdBuffers[frameIndex].bindPipeline(vk::PipelineBindPoint::eGraphics,*pipeline->pipeline);

					cmdBuffers[frameIndex].bindVertexBuffers(0, *(mesh.vertexBuffer), {0});
					cmdBuffers[frameIndex].bindIndexBuffer(*(mesh.indexBuffer), 0, vk::IndexType::eUint32);
					cmdBuffers[frameIndex].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->layout, 0, *object.descriptorSets[frameIndex], nullptr);
					cmdBuffers[frameIndex].drawIndexed(mesh.vertexCount, 1, 0, 0, 0);
				}
			}

			cmdBuffers[frameIndex].endRendering();
			
			// After rendering, transition the swapchain image to PRESENT_SRC
			transitionImageLayout(
					swapChainImages[imageIndex],
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::ImageLayout::ePresentSrcKHR,
					vk::AccessFlagBits2::eColorAttachmentWrite,             // srcAccessMask
					{},                                                     // dstAccessMask
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,     // srcStage
					vk::PipelineStageFlagBits2::eBottomOfPipe,               // dstStage
					vk::ImageAspectFlagBits::eColor
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
						msaaSamples = getMaxUsableSampleCount();
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
			LOG("Validation layer: type " << to_string(type) << "\n\tMessage: " << pCallbackData->pMessage)

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
			SDL_HideCursor();
			while(window->isRunning())
			{
				const bool *keys = SDL_GetKeyboardState(NULL);
				// TEMPORARY workaround before moving to my engine which actually handles input events properly
				float velocity = 0.0005f;
				auto front = camera->getFront(),
					 right = camera->getRight(),
					 up = camera->getUp();
				if(keys[SDL_SCANCODE_LSHIFT]){
					velocity = 0.001f;
				}
				if(keys[SDL_SCANCODE_W]){
					camera->pos += velocity * front;
				}
				if(keys[SDL_SCANCODE_S]){
					camera->pos -= velocity * front;
				}
				if(keys[SDL_SCANCODE_D]){
					camera->pos += velocity * right;
				}
				if(keys[SDL_SCANCODE_A]){
					camera->pos -= velocity * right;
				}
				if(keys[SDL_SCANCODE_SPACE]){
					camera->pos += velocity * up;
				}
				if(keys[SDL_SCANCODE_LCTRL]){
					camera->pos -= velocity * up;
				}
				while(!window->forwarded.empty())
				{
					auto [dx, dy] = window->forwarded.front();
					camera->yaw   -= dx * .1f;
					camera->pitch -= dy * .1f;
					camera->pitch = glm::clamp(camera->pitch, -89.0f, 89.0f);

					window->forwarded.pop();
				}
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
