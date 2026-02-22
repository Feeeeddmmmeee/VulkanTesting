#pragma once

#include "Window.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class GLFWWindow : public Window
{
	public:
		GLFWWindow(const WindowSpec &spec);
		~GLFWWindow();

		virtual bool createSurface(VkInstance instance, VkSurfaceKHR *surface) override;
		virtual void pollEvents() override;
		virtual std::pair<int,int> getFrameBufferSize() override;

	private:
		GLFWwindow *window;
};
