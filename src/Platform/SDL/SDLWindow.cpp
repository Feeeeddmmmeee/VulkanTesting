#include "SDLWindow.h"

#include <SDL3/SDL_vulkan.h>

Window *Window::create(const WindowSpec &spec)
{
	return new SDLWindow(spec);
}

const char* const *Window::getInstanceExtensions(uint32_t *count)
{
	return SDL_Vulkan_GetInstanceExtensions(count);
}

SDLWindow::SDLWindow(const WindowSpec &spec)
{
	this->spec = spec;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Vulkan_LoadLibrary(NULL);

	uint32_t flags = SDL_WINDOW_VULKAN|SDL_WINDOW_RESIZABLE;
	
	this->window = SDL_CreateWindow(spec.name.c_str(), spec.width, spec.height, flags);
	running = 1;
}

SDLWindow::~SDLWindow()
{
	SDL_DestroyWindowSurface(this->window);
	SDL_QuitSubSystem(SDL_INIT_VIDEO);
	SDL_Quit();
}

void SDLWindow::pollEvents()
{
	SDL_Event event;
	while(SDL_PollEvent(&event))
	{
		switch(event.type)
		{
			case SDL_EVENT_QUIT:
				running = 0;
				break;
			case SDL_EVENT_WINDOW_RESIZED:
				*(this->spec.frameBufferResizeCallback) = true;
				this->spec.width = event.window.data1;
				this->spec.height = event.window.data2;
				break;
		}
	}
}

bool SDLWindow::createSurface(VkInstance instance, VkSurfaceKHR *surface)
{
	return !SDL_Vulkan_CreateSurface(this->window, instance, nullptr, surface);
}
