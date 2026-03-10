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
	glm::vec3 worldUp={0,0,1};
	
	Camera(float w, float h, float fov=45.0f, glm::vec3 pos={0,0,0}, glm::vec3 front={1,0,0}, float near=.01f, float far=100.0f) : 
		width(w), height(h), fov(fov), pos(pos), near(near), far(far) {
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
		glm::mat4 view = lookAt(pos, pos+getFront(), getUp());

		return view;
	}
	
	glm::vec3 getFront()
	{
		glm::vec3 front;
		front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		front.y = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		front.z = sin(glm::radians(pitch));
		return glm::normalize(front);
	}

	glm::vec3 getRight()
	{
		return glm::normalize(glm::cross(getFront(), worldUp));
	}

	glm::vec3 getUp()
	{
		return glm::normalize(glm::cross(getRight(), getFront()));
	}
};
