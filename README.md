# VulkanTesting
This is a repostitory mostly intended for private use and reference by me when writing Vulkan code. As of the time of writing the project can use either SDL or GLFW to render using Vulkan's C++ RAII implementation.

<br>

![vulkan](https://github.com/user-attachments/assets/4148be7c-a53d-4dd8-9a2f-c29728808529)

<br>
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8652d246-7a1b-441a-b40e-6dcdc2864184" />

## :sparkles: Setup and configuration
After having installed all the [Dependencies](#package-dependencies), clone the repository. To run the demo execute the run script with the window backend of your choice (leave empty for cache/SDL by default).
```sh
git clone git@github.com:Feeeeddmmmeee/VulkanTesting.git
cd VulkanTesting
./run glfw
```
## :package: Dependencies
| Library | Use |
| --- | --- |
| SDL3 or GLFW | Window creation (only one of these is required) |
| Vulkan | Graphics API |
| LunarG's Vulkan SDK | Provides Vulkan validation layers + a slang shader compiler, though if you remove the _DEBUG definition from the Cmake file and compile the shaders yourself it is not needed. |
| stb_image | Header only image loading library |

<br>

<p align="center">
    <img src="https://github.com/catppuccin/catppuccin/blob/main/assets/footers/gray0_ctp_on_line.png?raw=true">
</p>
