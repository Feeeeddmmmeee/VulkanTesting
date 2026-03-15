#include "Texture.h"

void Texture::createSampler()
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

	sampler = vk::raii::Sampler(device, samplerInfo);
}

void VulkanImage::createImageView(vk::Format format, vk::ImageAspectFlags aspect, vk::raii::Device &device)
{
	// createImageView
	vk::ImageViewCreateInfo viewInfo{ .image = image, .viewType = vk::ImageViewType::e2D,
		.format = format, .subresourceRange = { aspect, 0, mipLevels, 0, 1 } };

	view = vk::raii::ImageView( device, viewInfo );
}

void Texture::createImageView()
{
	// createImageView
	vk::ImageViewCreateInfo viewInfo{ .image = image.image, .viewType = vk::ImageViewType::e2D,
		.format = vk::Format::eR8G8B8A8Srgb, .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, image.mipLevels, 0, 1 } };

	image.createImageView(vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, device);
}
