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
		virtual std::pair<int,int> getFrameBufferSize() override;

	private:
		SDL_Window *window;
		bool mouseHidden = true;

		void toggleMouse();
};
