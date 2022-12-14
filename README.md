# vulkan-tutorial-zig

A Zig implementation of [Vulkan Tutorial](https://vulkan-tutorial.com/) using [vulkan-zig](https://github.com/Snektron/vulkan-zig).

All code strives to match the C++ implementation as close as possible while making code more idiomatic to Zig. Implementation differences should be noted in this tutorial as to why the Zig version differs from the original.

Zig is still unstable. This repo strives to use `master` versions of Zig as breaking changes are introduced. Mileage will vary, but I will try to indicate versions of tools in the [development environment](#development-environment) section.

---
- [Introduction](#introduction)
- [Overview](#overview)
- [Development Environment](#development-environment)
- [Drawing a Triangle](#drawing-a-triangle)
  - [Setup](#setup)
    - [#00 Base Code](#base-code)
        - [General Structure](#general-structure)
        - [Resource Management](#resource-management)
        - [Integrating GLFW](#integrating-glfw)
    - [#01 Instance](#instance)
    - [#02 Validation Layers](#validation-layers)
    - [#03 Physical Devices and Queue Families](#physical-devices-and-queue-families)
    - [#04 Logical Device and Queues](#logical-device-and-queues)
  - [Presentation](#presentation)
    - [#05 Window Surface](#window-surface)
    - [#06 Swap Chain](#swap-chain)
    - [#07 Image Views](#image-views)
  - [Graphics Pipeline Basics](#graphics-pipeline-basics)
    - [#08 Introduction](#gp-introduction)
    - [#09 Shader Modules](#shader-modules)
    - [#10 Fixed Functions](#fixed-functions)
    - [#11 Render Passes](#render-passes)
    - [#12 Conclusion](#conclusion)
  - [Drawing](#drawing)
    - [#13 Framebuffers](#framebuffers)
    - [#14 Command Buffers](#command-buffers)
    - [#15 Rendering and Presentation](#rendering-and-presentation)
    - [#16 Frames in Flight](#frames-in-flight)
  - [#17 Swapchain Recreation](#swapchain-recreation)
- [Vertex Buffers](#vertex-buffers)
  - [#18 Vertex Input Description](#vertex-input-description)
  - [#19 Vertex Buffer Creation](#vertex-buffer-creation)
  - [#20 Staging Buffer](#staging-buffer)
  - [#21 Index Buffer](#index-buffer)
- [Uniform Buffers](#uniform-buffers)
  - [#22 Descriptor Layout and Buffer](#descriptor-layout-and-buffer)
  - [#23 Descriptor Pool and Sets](#descriptor-pool-and-sets)
- [Texture Mapping](#texture-mapping)
  - [#24 Images](#images)
  - [#25 Image View and Sampler](#image-view-and-sampler)
  - [#26 Combined Image Sampler](#combined-image-sampler)
- [#27 Depth Buffering](#depth-buffering)
- [#28 Loading Models](#loading-models)
- [#29 Generating Mipmaps](#generating-mipmaps)
- [#30 Multisampling](#multisampling)

# Introduction
This tutorial's README follows a similar structure to [bwasty/vulkan-tutorial-rs](https://github.com/bwasty/vulkan-tutorial-rs) and aims to fullfill the same intent. Chapters in this tutorial will go over only the differences between the Zig implementation and the C++ source content. The source [tutorial](https://vulkan-tutorial.com/) should still be able to provide the necessary knowledge to following along here.

The styles between the source material and this will differ based on Zig's general [style guide](https://ziglang.org/documentation/master/#Style-Guide) found in their documentation. This mostly affects camelCaseVars becoming snake_case_vars.

<span id="overview"></span>
# Overview : [tutorial](https://vulkan-tutorial.com/Overview)

<span id="development-environment"></span>
# Development Environment : [tutorial](https://vulkan-tutorial.com/Development_environment)

These binaries are needed on your system to operate this repo and run examples.

- Git
- Zig (stage2): [0.10.x](https://ziglang.org/download/)
- Zigmod: [r80+](https://github.com/nektro/zigmod/releases)
- Vulkan SDK: [latest](https://vulkan.lunarg.com/sdk/home)

## Running Examples
To initialize the repo with dependencies, run this in the root repo dir:

```
zigmod fetch
```

Running specific examples require the example's number, which is found in the source files of Vulkan tutorial (not the website). This README will label a given tutorial to a number. Zig build system's run step has a naming convention of `zig build run-XX`. Here's how we can run example `01`:

```
zig build run-01
```

## Individual Workspace

If you are working in your own workspace, I recommend using zigmod to dance around git clones/submodules dependencies we will need, but it is not required. The Zig build system only needs to know where to look for specific files/dirs. This tutorial will proceed with the assumption you are using Zigmod.

Create a `zigmod.yml` file along side your `build.zig` file with the contents of:
```yml
name: vulkan-tutorial-zig
build_dependencies:
- src: git https://github.com/Snektron/vulkan-zig
  main: generator/index.zig
  name: vk_gen
```

To get this repo pulled, run `zigmod fetch`.

This repo is a generator for Zig Vulkan bindings, so for us to be able to use this library, we will want to have it be generated at build time and referenced like so:

```zig
const deps = @import("deps.zig");
const vkgen = deps.imports.vk_gen;
...

const gen = vkgen.VkGenerateStep.init(b, deps.cache ++ "/git/github.com/Snektron/vulkan-zig/examples/vk.xml", "vk.zig");
exe.addPackage(gen.package);
```

This generator is looking for a Vulkan xml file, if you have one you would prefer to use, you can reference that one instead. For now, we will use the xml file found as part of the repo's example.

# Drawing a Triangle
## Setup
---
<span id="base-code"></span>
## #00 Base Code : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Base_code) | [code](src/00_base_code.zig)

### General Structure

```zig
const HelloTriangleApplication = struct {
    const Self = @This();

    pub fn init() Self {
        return Self{};
    }

    pub fn run(self: *Self) !void {
        try self.mainLoop();
    }

    fn mainLoop(self: *Self) !void {

    }

    pub fn deinit(self: *Self) void {

    }
};

pub fn main() anyerror!void {
    var app = HelloTriangleApplication.init();
    defer app.deinit();
    try app.run()
}
```

### Resource Management
Looking at Zig's std library, it is common practice to require an `init`/`deinit` function pair for the creation of a struct that could use an allocator or needs cleanup. While our program at this moment doesn't need an allocator, we will be using one for later examples. Just as the tutorial mentions, we will be manually cleaning up our Vulkan resources and placing them in the `deinit` (cleanup) function.

Where applicable, this tutorial will be setting our Vulkan fields to `.null_handle`, which is a 0 value for the various enum types this Vulkan binding provides. I prefer to know which resources have been set and clean them up as needed instead of blindly running the functions and potentially running into errors.

### Integrating GLFW
For our windowing, we will use a GLFW dependency. [Hexops' Zig bindings](https://github.com/hexops/mach-glfw) will do great for us here as it provides us with a more idiomatic API with error handling.

Add this src to your `zigmod.yml`:
```yml
build_dependencies:
- src: git https://github.com/hexops/mach-glfw
  main: build.zig
  name: build_glfw
```

This binding's README requires us to link this at build time, so we are adding it in with the `build_dependencies` section of the zigmod config. To let Zig know that this is a library we can use, we will need to add the package to the `exe` step in `build.zig` like so:

```zig
const glfw = deps.imports.build_glfw;
...

exe.addPackage(glfw.pkg);
glfw.link(b, exe, .{});
```

Using this library is a matter of adding the import and running the same code we see in the tutorial, but with a slight change to both the field type (now nullable) and how the functions are referenced.
```zig
const glfw = @import("glfw");

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

...
window: ?glfw.Window = null,
...

try glfw.init(.{});
self.window = try glfw.Window.create(WIDTH, HEIGHT, "Vulkan", null, null, .{
  .client_api = .no_api,
  .resizable = false,
});

...

while (!self.window.?.shouldClose()) {
    try glfw.pollEvents();
}
```

These should be all of the glfw functions needed to mimic exactly what the base code tutorial uses.

<span id="instance"></span>
## #01 Instance : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance) | [code](src/01_instance_creation.zig)

Now that we are using Vulkan in our example, we should go ahead and import the library like so `
const vk = @import("vulkan");`. This library doesn't load all of the function pointers that Vulkan can provide, so we will need to create dispatch wrappers with the specific functions to include:
```zig
const BaseDispatch = vk.BaseWrapper(.{
  .createInstance = true,
});

const InstanceDispatch = vk.InstanceWrapper(.{
  .destroyInstance = true,
});
```

We will be adding more dispatch flags as we progress through the tutorial. Reference the zig example code that should be linked next to every numbered tutorial. If you really don't care to add these as you need them, feel free to copy the dispatch wrappers from the final tutorial code. Earlier examples use `.cmdDraw` vs `.cmdDrawIndexed`.

We will be adding these fields to our app as `undefined`. Ideally, we don't reference these without acquiring the proc addresses ahead of time. If we wanted to be bit more safe here, we could make these nullable and set them to null, however to reference the dispatcher functions, it would be prefixed like so everytime `self.vki.?.myVkFunction()`.
```zig
vkb: BaseDispatch = undefined,
vki: InstanceDispatch = undefined,

...

// adding dispatches
const vk_proc = @ptrCast(fn (instance: vk.Instance, procname: [*:0]const u8) callconv(.C) vk.PfnVoidFunction, glfw.getInstanceProcAddress);
self.vkb = try BaseDispatch.load(vk_proc);

...

self.vki = try InstanceDispatch.load(self.instance, vk_proc);
```

While `@ptrCast` should not be used too often as stated [here](https://ziglang.org/documentation/master/#volatile), but in this particular case, I don't see a better alternative. Other examples will also be utilizing `@ptrCast` when we need to tell the Vulkan dispatch calls to only care about 1 element objects that are not initialized as an array and converting said object into a sentinel array of the same Type.

For the sake of making the Zig and C++ implementations mirror each other as best as possible, this tutorial will be creating consts needed for various Vulkan functions with matching variable names. Throughout the tutorial, it may be interchanged with `.{}` instead of `vk.Object{}` as Zig's compiler is able to infer what type that struct ought to be. Here is how it would look between the two styles:
```zig
const create_info = vk.InstanceCreateInfo{
    .flags = .{},
    ...
};

self.instance = try self.vkb.createInstance(&create_info, null);
```
```zig
self.instance = try self.vkb.createInstance(&.{
  .flags = .{},
  ...
}, null);
```

<span id="validation-layers"></span>
## #02 Validation Layers : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers) | [code](src/02_validation_layers.zig)

We will need to introduce the use of an allocator for this example and beyond. To ensure we cleanup all of the memory allocated, we will log an error message on the allocator's cleanup and indicate any memory leak.

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
  defer {
    const leaked = gpa.deinit();
    if (leaked) std.log.err("MemLeak", .{});
  }
  const allocator = gpa.allocator();

  var app = HelloTriangleApplication.init(allocator);
  defer app.deinit();
  app.run() catch |err| {
    std.log.err("application exited with error: {any}", .{err});
    return;
  };
```

You may notice that `main` is no longer returning an `anyerror!void`. One of my workstations uses a Windows OS and returning an error from main made the output straining to parse, so we log any errors returned from the app.

Adding a field to hold the allocator handle to our application is typical for any struct that needs an allocator to reference within its functions.
```
const Allocator = std.mem.Allocator;
...

const HelloTriangleApplication = struct {
  const Self = @This();
  allocator: Allocator,
  ...
  pub fn init(allocator: Allocator) Self {
    return Self{ .allocator = allocator };
  }
  ...
};
```

Debug vs Release mode can be known at comptime and we can create a bool of whether we are in a debug mode like so:
```zig
const builtin = @import("builtin");

const enable_validation_layers: bool = switch (builtin.mode) {
  .Debug, .ReleaseSafe => true,
  else => false,
};
```

<span id="physical-devices-and-queue-families"></span>
## #03 Physical Devices and Queue Families : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families) | [code](src/03_physical_device_selection.zig)

<span id="logical-device-and-queues"></span>
## #04 Logical Device and Queues : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues) | [code](src/04_logical_device.zig)

<span id="presentation"></span>
## Presentation
---
<span id="window-surface"></span>
## #05 Window Surface : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface) | [code](src/05_window_surface.zig)

If we setup our `findQueueFamilies` function to mimic the C++ 1:1, we might end up with the following error:
```
debug: validation layer: Validation Error: [ VUID-VkDeviceCreateInfo-queueFamilyIndex-02802 ] Object 0: handle = 0x23372053850, type = VK_OBJECT_TYPE_PHYSICAL_DEVICE; | MessageID = 0x29498778 | CreateDevice(): pCreateInfo->pQueueCreateInfos[1].queueFamilyIndex (=0) is not unique and was also used in pCreateInfo->pQueueCreateInfos[0]. The Vulkan spec states: The queueFamilyIndex member of each element of pQueueCreateInfos must be unique within pQueueCreateInfos, except that two members can share the same queueFamilyIndex if one describes protected-capable queues and one describes queues that are not protected-capable (https://vulkan.lunarg.com/doc/view/1.3.216.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-queueFamilyIndex-02802)
```

To avoid this, I placed the indices check in an `if`/`else if` set of conditionals. The Vulkan example ought to run all the same if it were left alone, but we would be throwing Vulkan validation layer errors.

<span id="swap-chain"></span>
## #06 Swap Chain : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain) | [code](src/06_swap_chain_creation.zig)

<span id="image-views"></span>
## #07 Image Views : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views) | [code](src/07_image_views.zig)

## Graphics Pipeline Basics
---
<span id="gp-introduction"></span>
## #08 Introduction : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Introduction) | [code](src/08_graphics_pipeline.zig)

<span id="shader-modules"></span>
## #09 Shader Modules : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules) | [code](src/09_shader_module.zig)

The vulkan-zig repo comes with 2 different zig files we can import and utilize. The build.zig file should already be generating and linking the Vulkan package at build time. The vulkan repo also has some util functions in its build.zig file which we will be using to convert frag/vert shader files to sprv at build time with glslc. These shader files will also be embedded as part of the binary to be read from the `resources.zig` file (located in zig-cache) the shader package provides from the util function. If you wish to just generate or use your own sprv files, you will want to tweak the tutorial to do so.

First we will need to add vulkan-zig again as a build_dependency, but with a different name and main:
```yml
- src: git https://github.com/Snektron/vulkan-zig
  main: build.zig
  name: vk_build
```

To start adding shaders at build time, add similar lines to your build.zig:

```zig
const shaders = vkbuild.ResourceGenStep.init(b, "resources.zig");
shaders.addShader("vert", "src/09_shader_base.vert");
shaders.addShader("frag", "src/09_shader_base.frag");
exe.addPackage(shaders.package);
```

We can reference these shaders from our code by importing resources:
```zig
const resources = @import("resources");

...

const vert_shader_module: vk.ShaderModule = try self.createShaderModule(resources.vert);
defer self.vkd.destroyShaderModule(self.device, vert_shader_module, null);
const frag_shader_module: vk.ShaderModule = try self.createShaderModule(resources.frag);
defer self.vkd.destroyShaderModule(self.device, frag_shader_module, null);
```

To reiterate, this is just one way to go about using shaders. You can have them built at build time, comptime, and read them at runtime. Please use the method that best suits your projects' needs.

<span id="fixed-functions"></span>
## #10 Fixed Functions : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions) | [code](src/10_fixed_functions.zig)

<span id="render-passes"></span>
## #11 Render Passes : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes) | [code](src/11_render_passes.zig)

<span id="conclusion"></span>
## #12 Conclusion : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Conclusion) | [code](src/12_graphics_pipeline_complete.zig)

## Drawing
---
<span id="framebuffers"></span>
## #13 Framebuffers : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Framebuffers) | [code](src/13_framebuffers.zig)

<span id="command-buffers"></span>
## #14 Command Buffers : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers) | [code](src/14_command_buffers.zig)

<span id="rendering-and-presentation"></span>
## #15 Rendering and Presentation : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Rendering_and_presentation) | [code](src/15_hello_triangle.zig)

<span id="frames-in-flight"></span>
## #16 Frames in Flight : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Frames_in_flight) | [code](src/16_frames_in_flight.zig)

<span id="swapchain-recreation"></span>
## #17 Swapchain Recreation : [tutorial](https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation) | [code](src/17_swap_chain_recreation.zig)

The vulkan-zig generated package will emit errors from most all dispatches. This is usually great for idiomatic Zig coding, but in this particular case, it will make catching and working with errors a little wonky to read at times. The following code is needed to catch the `VK_ERROR_OUT_OF_DATE_KHR` result from `queuePresentKHR`. It catches all errors, and specifically on the error we wish to "ignore", we will set the result to the enum value, otherwise return the error.

```zig
const present_result = self.vkd.queuePresentKHR(self.present_queue, &.{
    .wait_semaphore_count = signal_semaphores.len,
    .p_wait_semaphores = @ptrCast([*]const vk.Semaphore, &signal_semaphores),
    .swapchain_count = 1,
    .p_swapchains = @ptrCast([*]const vk.SwapchainKHR, &self.swap_chain),
    .p_image_indices = @ptrCast([*]const u32, &result.image_index),
    .p_results = null,
}) catch |err| switch (err) {
    error.OutOfDateKHR => vk.Result.error_out_of_date_khr,
    else => return err,
};

if (present_result == .error_out_of_date_khr or present_result == .suboptimal_khr or self.framebuffer_resized) {
    self.framebuffer_resized = false;
    try self.recreateSwapChain();
} else if (present_result != .success) {
    return error.ImagePresentFailed;
}
```

# Vertex Buffers
<span id="vertex-input-description"></span>
## #18 Vertex Input Description : [tutorial](https://vulkan-tutorial.com/Vertex_buffers/Vertex_input_description) | [code](src/18_vertex_input.zig)

### Expected Result

The window will be blank.

You may come across these validation layer errors upon completing this example. Don't worry about these too much, they should be resolved in the next example. Vulkan is upset to be told about vertex inputs with no buffers supplying data.
```
debug: validation layer: Validation Error: [ VUID-vkCmdDraw-None-02721 ] Object 0: handle = 0x255d971fc80, type = VK_OBJECT_TYPE_COMMAND_BUFFER; Object 1: handle = 0x967dd1000000000e, type = VK_OBJECT_TYPE_PIPELINE; | MessageID = 0x99ef63bb | vkCmdDraw: binding #0 in pVertexAttributeDescriptions[1] of VkPipeline 0x967dd1000000000e[] is an invalid value for command buffer VkCommandBuffer 0x255d971fc80[]. The Vulkan spec states: For a given vertex buffer binding, any attribute data fetched must be entirely contained within the corresponding vertex buffer binding, as described in Vertex Input Description (https://vulkan.lunarg.com/doc/view/1.3.216.0/windows/1.3-extensions/vkspec.html#VUID-vkCmdDraw-None-02721)

debug: validation layer: Validation Error: [ VUID-vkCmdDraw-None-04007 ] Object 0: handle = 0x255d971fc80, type = VK_OBJECT_TYPE_COMMAND_BUFFER; | MessageID = 0x9981c31b | vkCmdDraw: VkPipeline 0x967dd1000000000e[] expects that this Command Buffer's vertex binding Index 0 should be set via vkCmdBindVertexBuffers. This is because pVertexBindingDescriptions[0].binding value is 0. The Vulkan spec states: All vertex input bindings accessed via vertex input variables declared in the vertex shader entry point's interface must have either valid or VK_NULL_HANDLE buffers bound (https://vulkan.lunarg.com/doc/view/1.3.216.0/windows/1.3-extensions/vkspec.html#VUID-vkCmdDraw-None-04007)
```

<span id="vertex-buffer-creation"></span>
## #19 Vertex Buffer Creation : [tutorial](https://vulkan-tutorial.com/Vertex_buffers/Vertex_buffer_creation) | [code](src/19_vertex_buffer.zig)

<span id="staging-buffer"></span>
## #20 Staging Buffer : [tutorial](https://vulkan-tutorial.com/Vertex_buffers/Staging_buffer) | [code](src/20_staging_buffer.zig)

<span id="index-buffer"></span>
## #21 Index Buffer : [tutorial](https://vulkan-tutorial.com/Vertex_buffers/Index_buffer) | [code](src/21_index_buffer.zig)

# Uniform Buffers
<span id="descriptor-layout-and-buffer"></span>
## #22 Descriptor Layout and Buffer : [tutorial](https://vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer) | [code](src/22_descriptor_layout.zig)

### Expected Result

The window will be blank.

This will be resolved in the next example. Here are some example Vulkan validation layer errors that will show up:
```
debug: validation layer: Validation Error: [ VUID-vkCmdDrawIndexed-None-02697 ] Object 0: handle = 0xe7e6d0000000000f, type = VK_OBJECT_TYPE_PIPELINE; Object 1: handle = 0x967dd1000000000e, type = VK_OBJECT_TYPE_PIPELINE_LAYOUT; Object 2: VK_NULL_HANDLE, type = VK_OBJECT_TYPE_PIPELINE_LAYOUT; | MessageID = 0x9888fef3 | vkCmdDrawIndexed(): VkPipeline 0xe7e6d0000000000f[] defined with VkPipelineLayout 0x967dd1000000000e[] is not compatible for maximum set statically used 0 with bound descriptor sets, last bound with VkPipelineLayout 0x0[] The Vulkan spec states: For each set n that is statically used by the VkPipeline bound to the pipeline bind point used by this command, a descriptor set must have been bound to n at the same pipeline bind point, with a VkPipelineLayout that is compatible for set n, with the VkPipelineLayout used to create the current VkPipeline, as described in Pipeline Layout Compatibility (https://vulkan.lunarg.com/doc/view/1.3.216.0/windows/1.3-extensions/vkspec.html#VUID-vkCmdDrawIndexed-None-02697)
debug: validation layer: Validation Error: [ UNASSIGNED-CoreValidation-DrawState-DescriptorSetNotBound ] Object 0: handle = 0x1bfedf77850, type = VK_OBJECT_TYPE_COMMAND_BUFFER; | MessageID = 0xcde11083 | vkCmdDrawIndexed(): VkPipeline 0xe7e6d0000000000f[] uses set #0 but that set is not bound.
```


### A New Dependency

We are introducing a linear algebra library to be used. I picked [kooparse/zalgebra](https://github.com/kooparse/zalgebra) as it had a friendly api, but you can choose to swap it out with any linear algebra library you wish to choose such as [ziglibs/zlm](https://github.com/ziglibs/zlm). Here's how we will need to add it to our dependencies.

```yml
root_dependencies:
- src: git https://github.com/kooparse/zalgebra
```

Unlike the glfw and Vulkan libraries, we will need to add this to the `root_dependencies` block for us to reference in our code. Don't forget to run `zigmod fetch` after updating the yml file, if you are using zigmod.

To link it, we can add this code to our `build.zig`:
```zig
deps.addAllTo(exe);
```

Now for our code to use this library, it's just an import away:
```zig
const za = @import("zalgebra");
```

<span id="descriptor-pool-and-sets"></span>
## #23 Descriptor Pool and Sets : [tutorial](https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets) | [code](src/23_descriptor_sets.zig)
# Texture Mapping
<span id="images"></span>
## #24 Images : [tutorial](https://vulkan-tutorial.com/Texture_mapping/Images) | [code](src/24_texture_image.zig)

For our texture image, we will use the same image provided in the source tutorial that is this [CC0 licensed image](https://pixabay.com/photos/statue-sculpture-figure-1275469/) resized to 512x512.

### A New Dependency

At the time of writing this, I wasn't able to utilize any Zig image loading libraries with the images this tutorial uses, so we will be using the same library the C++ tutorial recommends, which is written in C as a precompiled header. To use this header, we will need to make some additions to our `build.zig`.

First as a hack, we can add [nothings/stb](https://github.com/nothings/stb) as a git reference in our `zigmod.yml` under `build_dependencies`:

```yml
- src: git https://github.com/nothings/stb
  name: stb
  main: ''
```

Now this isn't *correct* for zigmod to be used in this way, but we aren't going to be directly referencing this repo as a package in our build file. You can directly download the `stb_image.h` file yourself if you want. I didn't want to drop this file in the repo if it wasn't needed.

We need to let Zig know about this header file and to do so, we will add the include dir from the `.zigmod` dir:
```zig
exe.addIncludeDir(deps.cache ++ "/git/github.com/nothings/stb");
```

We aren't done with making modifications to the `build.zig` file just yet, we need to link C and inform Zig how to compile this file. Zig does not let us import headers directly and use it as needed as indicated by this [issue](https://github.com/ziglang/zig/issues/3495). We will need to create a C file that will define the necessary constant and include the header.

```c
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
```

And add it to `build.zig`:
```zig
exe.linkLibC();
exe.addCSourceFile("libs/stb/stb_impl.c", &.{"-std=c99"});
```

To be used in our code, we need to import it:
```zig
const c = @cImport({
    @cInclude("stb_image.h");
});
```

Any use of this library needs to be prefixed with `c.`:
```zig
const pixels = c.stbi_load("resources/texture.jpg", &tex_width, &tex_height, &channels, c.STBI_rgb_alpha);
```

For `@cImport` best practices, refer to the [docs](https://ziglang.org/documentation/master/#cImport).

<span id="image-view-and-sampler"></span>
## #25 Image View and Sampler : [tutorial](https://vulkan-tutorial.com/Texture_mapping/Image_view_and_sampler) | [code](src/25_sampler.zig)

<span id="combined-image-sampler"></span>
## #26 Combined Image Sampler : [tutorial](https://vulkan-tutorial.com/Texture_mapping/Combined_image_sampler) | [code](src/26_texture_mapping.zig)

<span id="depth-buffering"></span>
# #27 Depth Buffering : [tutorial](https://vulkan-tutorial.com/Depth_buffering) | [code](src/27_depth_buffering.zig)

<span id="loading-models"></span>
# #28 Loading Models : [tutorial](https://vulkan-tutorial.com/Loading_models) | [code](src/28_model_loading.zig)

For our model, we will be using the same [Viking room](https://sketchfab.com/3d-models/viking-room-a49f1b8e4f5c4ecf9e1fe7d81915ad38) model by [nigelgoh](https://sketchfab.com/nigelgoh) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) found in the source tutorial.

### A New Dependency

```yml
root_dependencies:
- src: git https://github.com/ziglibs/wavefront-obj
  name: wavefront-obj
  main: wavefront-obj.zig
```

We have to provide overrides for `name` and `main` as the zigmod.yml for that repo is improperly telling us how to ingest it. In cases like this, we could create a PR to fix it for others, but for now we can overwrite it ourselves to not be stuck.

<span id="generating-mipmaps"></span>
# #29 Generating Mipmaps : [tutorial](https://vulkan-tutorial.com/Generating_Mipmaps) | [code](src/29_mipmapping.zig)

<span id="multisampling"></span>
# #30 Multisampling : [tutorial](https://vulkan-tutorial.com/Multisampling) | [code](src/30_multisampling.zig)
