#pragma once

#include "Window.h"

#include <SDL3/SDL.h>

class SDLWindow : public Window
{
	public:
		SDLWindow(const WindowSpec &spec);
		~SDLWindow();

		virtual bool createSurface(VkInstance instance, VkSurfaceKHR *surface) override;
		virtual void pollEvents() override;

	private:
		SDL_Window *window;
};
