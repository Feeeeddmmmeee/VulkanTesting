#pragma once

#include <string>
#include <utility>

#include <vulkan/vulkan.h>

struct WindowSpec
{
	std::string name;
	uint32_t width, height;
	bool *frameBufferResizeCallback;
};

class Window
{
	public:
		virtual ~Window() = default;
		virtual bool createSurface(VkInstance instance, VkSurfaceKHR *surface) = 0;
		virtual void pollEvents() = 0;
		virtual std::pair<int,int> getFrameBufferSize() = 0;

		const WindowSpec &getSpec() { return this->spec; }
		bool isRunning() { return this->running; }

		static Window *create(const WindowSpec &spec);
		static const char* const* getInstanceExtensions(uint32_t *count);

	protected:
		WindowSpec spec;
		bool running;
};
