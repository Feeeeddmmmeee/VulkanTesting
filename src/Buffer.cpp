#include "Buffer.h"

VulkanBuffer::VulkanBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags memoryFlags, vk::raii::Device &dev, vk::raii::PhysicalDevice &pDev)
	: device(dev), pDevice(pDev), size(size)
{
	vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage,
		.sharingMode = vk::SharingMode::eExclusive };

	buffer = vk::raii::Buffer(device, bufferInfo);

	vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

	// findMemoryType
	uint32_t memType;
	vk::PhysicalDeviceMemoryProperties memProperties = pDevice.getMemoryProperties();
	for (int i = 0; i < memProperties.memoryTypeCount; ++i) {
		if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & memoryFlags) == memoryFlags) {
			memType = i;
			break;
		}
	}

	vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size, .memoryTypeIndex = memType };
	memory = vk::raii::DeviceMemory(device, allocInfo);
	buffer.bindMemory(*memory, 0);
}

void VulkanBuffer::uploadToMemory(void *src, int offset)
{
	memcpy((char*)data+offset, src, size);
}

void VulkanBuffer::mapMemory()
{
	data = memory.mapMemory(0, size);
}

void VulkanBuffer::unmapMemory()
{
	memory.unmapMemory();
}
