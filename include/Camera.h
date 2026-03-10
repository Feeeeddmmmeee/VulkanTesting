#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera
{
	float fov;
	glm::vec3 pos;
	float near, far;
	float width, height;

	float yaw, pitch;
	glm::vec3 front, worldUp={0,0,1};
	glm::vec3 right, up;
	
	Camera(float w, float h, float fov=45.0f, glm::vec3 pos={0,0,0}, glm::vec3 front={1,0,0}, float near=.01f, float far=100.0f) : 
		width(w), height(h), fov(fov), pos(pos), near(near), far(far), front(front) {
			front = glm::normalize(front);
			right = glm::normalize(glm::cross(front, worldUp));
			up = glm::normalize(glm::cross(right, front));

			pitch = glm::degrees(glm::asin(front.z));
			yaw = glm::degrees(atan2(front.y, front.x));
		}

	glm::mat4 getProjMatrix()
	{
		glm::mat4 proj = glm::perspective(glm::radians(fov), width/height, near, far);
		proj[1][1] *= -1; // otherwise it would be upside down

		return proj;
	}

	glm::mat4 getViewMatrix()
	{
		glm::mat4 view = lookAt(pos, pos+front, up);

		return view;
	}
};
