#include "GLFWWindow.h"
#include <GLFW/glfw3.h>

Window *Window::create(const WindowSpec &spec)
{
	return new GLFWWindow(spec);
}

const char* const *Window::getInstanceExtensions(uint32_t *count)
{
	return glfwGetRequiredInstanceExtensions(count);
}

GLFWWindow::GLFWWindow(const WindowSpec &spec)
{
	this->spec = spec;

	// Hyprland window dimension fix
	glfwInitHint(GLFW_WAYLAND_LIBDECOR, GLFW_WAYLAND_DISABLE_LIBDECOR);

	glfwInit();
	// Disable OpenGL
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(spec.width, spec.height, spec.name.c_str(), nullptr, nullptr);
	glfwSetWindowUserPointer(this->window, this->spec.frameBufferResizeCallback);
	glfwSetFramebufferSizeCallback(window, [](GLFWwindow *win, int width, int height) {
			*(reinterpret_cast<bool*>(glfwGetWindowUserPointer(win))) = true;
		});

	this->running = 1;
}

GLFWWindow::~GLFWWindow()
{
	glfwDestroyWindow(this->window);
	glfwTerminate();
}

void GLFWWindow::pollEvents()
{
	if(glfwWindowShouldClose(this->window)) this->running = 0;
	glfwPollEvents();
}

bool GLFWWindow::createSurface(VkInstance instance, VkSurfaceKHR *surface)
{
	return glfwCreateWindowSurface(instance, this->window, nullptr, surface);
}
std::pair<int,int> GLFWWindow::getFrameBufferSize()
{
	std::pair<int,int> out;
	glfwGetFramebufferSize(this->window,&out.first,&out.second);
	return out;
}
