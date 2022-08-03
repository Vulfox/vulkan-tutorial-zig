const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const glfw = @import("glfw");
const vk = @import("vulkan");
const resources = @import("resources");

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

const device_extensions = [_][*:0]const u8{vk.extension_info.khr_swapchain.name};

const enable_validation_layers: bool = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    else => false,
};

const BaseDispatch = vk.BaseWrapper(.{
    .createInstance = true,
    .enumerateInstanceLayerProperties = true,
});

const InstanceDispatch = vk.InstanceWrapper(.{
    .createDebugUtilsMessengerEXT = enable_validation_layers,
    .createDevice = true,
    .destroyDebugUtilsMessengerEXT = enable_validation_layers,
    .destroyInstance = true,
    .destroySurfaceKHR = true,
    .enumerateDeviceExtensionProperties = true,
    .enumeratePhysicalDevices = true,
    .getDeviceProcAddr = true,
    .getPhysicalDeviceMemoryProperties = true,
    .getPhysicalDeviceQueueFamilyProperties = true,
    .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
    .getPhysicalDeviceSurfaceFormatsKHR = true,
    .getPhysicalDeviceSurfacePresentModesKHR = true,
    .getPhysicalDeviceSurfaceSupportKHR = true,
});

const DeviceDispatch = vk.DeviceWrapper(.{
    .acquireNextImageKHR = true,
    .allocateCommandBuffers = true,
    .allocateMemory = true,
    .beginCommandBuffer = true,
    .bindBufferMemory = true,
    .cmdBeginRenderPass = true,
    .cmdBindPipeline = true,
    .cmdBindVertexBuffers = true,
    .cmdCopyBuffer = true,
    .cmdDraw = true,
    .cmdEndRenderPass = true,
    .cmdSetViewport = true,
    .cmdSetScissor = true,
    .createBuffer = true,
    .createCommandPool = true,
    .createFence = true,
    .createFramebuffer = true,
    .createGraphicsPipelines = true,
    .createImageView = true,
    .createPipelineLayout = true,
    .createRenderPass = true,
    .createSemaphore = true,
    .createShaderModule = true,
    .createSwapchainKHR = true,
    .destroyBuffer = true,
    .destroyCommandPool = true,
    .destroyDevice = true,
    .destroyFence = true,
    .destroyFramebuffer = true,
    .destroyImageView = true,
    .destroyPipeline = true,
    .destroyPipelineLayout = true,
    .destroyRenderPass = true,
    .destroySemaphore = true,
    .destroyShaderModule = true,
    .destroySwapchainKHR = true,
    .deviceWaitIdle = true,
    .endCommandBuffer = true,
    .freeCommandBuffers = true,
    .freeMemory = true,
    .getBufferMemoryRequirements = true,
    .getDeviceQueue = true,
    .getSwapchainImagesKHR = true,
    .mapMemory = true,
    .queuePresentKHR = true,
    .queueSubmit = true,
    .queueWaitIdle = true,
    .resetCommandBuffer = true,
    .resetFences = true,
    .unmapMemory = true,
    .waitForFences = true,
});

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    fn isComplete(self: *const QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

pub const SwapChainSupportDetails = struct {
    allocator: Allocator,
    capabilities: vk.SurfaceCapabilitiesKHR = undefined,
    formats: ?[]vk.SurfaceFormatKHR = null,
    present_modes: ?[]vk.PresentModeKHR = null,

    pub fn init(allocator: Allocator) SwapChainSupportDetails {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: SwapChainSupportDetails) void {
        if (self.formats != null) self.allocator.free(self.formats.?);
        if (self.present_modes != null) self.allocator.free(self.present_modes.?);
    }
};

pub const Vertex = struct {
    pos: [2]f32 = .{ 0, 0 },
    color: [3]f32 = .{ 0, 0, 0 },

    pub fn getBindingDescription() vk.VertexInputBindingDescription {
        return vk.VertexInputBindingDescription{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .input_rate = .vertex,
        };
    }

    pub fn getAttributeDescriptions() [2]vk.VertexInputAttributeDescription {
        return [2]vk.VertexInputAttributeDescription{
            .{
                .binding = 0,
                .location = 0,
                .format = .r32g32_sfloat,
                .offset = @offsetOf(Vertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = .r32g32b32_sfloat,
                .offset = @offsetOf(Vertex, "color"),
            },
        };
    }
};

const vertices = [_]Vertex{
    .{ .pos = .{ 0.0, -0.5 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 0.5, 0.5 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ -0.5, 0.5 }, .color = .{ 0, 0, 1 } },
};

const HelloTriangleApplication = struct {
    const Self = @This();
    allocator: Allocator,

    window: ?glfw.Window = null,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,
    vkd: DeviceDispatch = undefined,

    instance: vk.Instance = .null_handle,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null_handle,
    surface: vk.SurfaceKHR = .null_handle,

    physical_device: vk.PhysicalDevice = .null_handle,
    device: vk.Device = .null_handle,

    graphics_queue: vk.Queue = .null_handle,
    present_queue: vk.Queue = .null_handle,

    swap_chain: vk.SwapchainKHR = .null_handle,
    swap_chain_images: ?[]vk.Image = null,
    swap_chain_image_format: vk.Format = .@"undefined",
    swap_chain_extent: vk.Extent2D = .{ .width = 0, .height = 0 },
    swap_chain_image_views: ?[]vk.ImageView = null,
    swap_chain_framebuffers: ?[]vk.Framebuffer = null,

    render_pass: vk.RenderPass = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,
    graphics_pipeline: vk.Pipeline = .null_handle,

    command_pool: vk.CommandPool = .null_handle,

    vertex_buffer: vk.Buffer = .null_handle,
    vertex_buffer_memory: vk.DeviceMemory = .null_handle,

    command_buffers: ?[]vk.CommandBuffer = null,

    image_available_semaphores: ?[]vk.Semaphore = null,
    render_finished_semaphores: ?[]vk.Semaphore = null,
    in_flight_fences: ?[]vk.Fence = null,
    current_frame: u32 = 0,

    framebuffer_resized: bool = false,

    pub fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn run(self: *Self) !void {
        try self.initWindow();
        try self.initVulkan();
        try self.mainLoop();
    }

    fn initWindow(self: *Self) !void {
        try glfw.init(.{});
        self.window = try glfw.Window.create(WIDTH, HEIGHT, "Vulkan", null, null, .{
            .client_api = .no_api,
        });
        self.window.?.setUserPointer(self);
        self.window.?.setFramebufferSizeCallback(framebufferResizeCallback);
    }

    fn framebufferResizeCallback(window: glfw.Window, _: u32, _: u32) void {
        var self = window.getUserPointer(Self);
        if (self != null) {
            self.?.framebuffer_resized = true;
        }
    }

    fn initVulkan(self: *Self) !void {
        try self.createInstance();
        try self.setupDebugMessenger();
        try self.createSurface();
        try self.pickPhysicalDevice();
        try self.createLogicalDevice();
        try self.createSwapChain();
        try self.createImageViews();
        try self.createRenderPass();
        try self.createGraphicsPipeline();
        try self.createFramebuffers();
        try self.createCommandPool();
        try self.createVertexBuffer();
        try self.createCommandBuffers();
        try self.createSyncObjects();
    }

    fn mainLoop(self: *Self) !void {
        while (!self.window.?.shouldClose()) {
            try glfw.pollEvents();
            try self.drawFrame();
        }

        _ = try self.vkd.deviceWaitIdle(self.device);
    }

    fn cleanupSwapChain(self: *Self) void {
        if (self.swap_chain_framebuffers != null) {
            for (self.swap_chain_framebuffers.?) |framebuffer| {
                self.vkd.destroyFramebuffer(self.device, framebuffer, null);
            }
            self.allocator.free(self.swap_chain_framebuffers.?);
            self.swap_chain_framebuffers = null;
        }

        if (self.swap_chain_image_views != null) {
            for (self.swap_chain_image_views.?) |image_view| {
                self.vkd.destroyImageView(self.device, image_view, null);
            }
            self.allocator.free(self.swap_chain_image_views.?);
            self.swap_chain_image_views = null;
        }

        if (self.swap_chain_images != null) {
            self.allocator.free(self.swap_chain_images.?);
            self.swap_chain_images = null;
        }

        if (self.swap_chain != .null_handle) {
            self.vkd.destroySwapchainKHR(self.device, self.swap_chain, null);
            self.swap_chain = .null_handle;
        }
    }

    pub fn deinit(self: *Self) void {
        self.cleanupSwapChain();

        if (self.graphics_pipeline != .null_handle) self.vkd.destroyPipeline(self.device, self.graphics_pipeline, null);
        if (self.pipeline_layout != .null_handle) self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
        if (self.render_pass != .null_handle) self.vkd.destroyRenderPass(self.device, self.render_pass, null);

        if (self.vertex_buffer != .null_handle) self.vkd.destroyBuffer(self.device, self.vertex_buffer, null);
        if (self.vertex_buffer_memory != .null_handle) self.vkd.freeMemory(self.device, self.vertex_buffer_memory, null);

        if (self.render_finished_semaphores != null) {
            for (self.render_finished_semaphores.?) |semaphore| {
                self.vkd.destroySemaphore(self.device, semaphore, null);
            }
            self.allocator.free(self.render_finished_semaphores.?);
        }
        if (self.image_available_semaphores != null) {
            for (self.image_available_semaphores.?) |semaphore| {
                self.vkd.destroySemaphore(self.device, semaphore, null);
            }
            self.allocator.free(self.image_available_semaphores.?);
        }
        if (self.in_flight_fences != null) {
            for (self.in_flight_fences.?) |fence| {
                self.vkd.destroyFence(self.device, fence, null);
            }
            self.allocator.free(self.in_flight_fences.?);
        }

        if (self.command_pool != .null_handle) self.vkd.destroyCommandPool(self.device, self.command_pool, null);
        if (self.command_buffers != null) self.allocator.free(self.command_buffers.?);

        if (self.device != .null_handle) self.vkd.destroyDevice(self.device, null);

        if (enable_validation_layers and self.debug_messenger != .null_handle) self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);

        if (self.surface != .null_handle) self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        if (self.instance != .null_handle) self.vki.destroyInstance(self.instance, null);

        if (self.window != null) self.window.?.destroy();

        glfw.terminate();
    }

    fn recreateSwapChain(self: *Self) !void {
        var size = try glfw.Window.getFramebufferSize(self.window.?);

        while (size.width == 0 or size.height == 0) {
            size = try glfw.Window.getFramebufferSize(self.window.?);
            try glfw.waitEvents();
        }

        try self.vkd.deviceWaitIdle(self.device);

        self.cleanupSwapChain();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createFramebuffers();
    }

    fn createInstance(self: *Self) !void {
        const vk_proc = @ptrCast(fn (instance: vk.Instance, procname: [*:0]const u8) callconv(.C) vk.PfnVoidFunction, glfw.getInstanceProcAddress);
        self.vkb = try BaseDispatch.load(vk_proc);

        if (enable_validation_layers and !try self.checkValidationLayerSupport()) {
            return error.MissingValidationLayer;
        }

        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(1, 0, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0, 0),
            .api_version = vk.API_VERSION_1_2,
        };

        const extensions = try getRequiredExtensions(self.allocator);
        defer extensions.deinit();

        var create_info = vk.InstanceCreateInfo{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = @intCast(u32, extensions.items.len),
            .pp_enabled_extension_names = extensions.items.ptr,
        };

        if (enable_validation_layers) {
            create_info.enabled_layer_count = validation_layers.len;
            create_info.pp_enabled_layer_names = &validation_layers;

            var debug_create_info: vk.DebugUtilsMessengerCreateInfoEXT = undefined;
            populateDebugMessengerCreateInfo(&debug_create_info);
            create_info.p_next = &debug_create_info;
        }

        self.instance = try self.vkb.createInstance(&create_info, null);

        self.vki = try InstanceDispatch.load(self.instance, vk_proc);
    }

    fn populateDebugMessengerCreateInfo(create_info: *vk.DebugUtilsMessengerCreateInfoEXT) void {
        create_info.* = .{
            .flags = .{},
            .message_severity = .{
                .verbose_bit_ext = true,
                .warning_bit_ext = true,
                .error_bit_ext = true,
            },
            .message_type = .{
                .general_bit_ext = true,
                .validation_bit_ext = true,
                .performance_bit_ext = true,
            },
            .pfn_user_callback = debugCallback,
            .p_user_data = null,
        };
    }

    fn setupDebugMessenger(self: *Self) !void {
        if (!enable_validation_layers) return;

        var create_info: vk.DebugUtilsMessengerCreateInfoEXT = undefined;
        populateDebugMessengerCreateInfo(&create_info);

        self.debug_messenger = try self.vki.createDebugUtilsMessengerEXT(self.instance, &create_info, null);
    }

    fn createSurface(self: *Self) !void {
        if ((try glfw.createWindowSurface(self.instance, self.window.?, null, &self.surface)) != @enumToInt(vk.Result.success)) {
            return error.SurfaceInitFailed;
        }
    }

    fn pickPhysicalDevice(self: *Self) !void {
        var device_count: u32 = undefined;
        _ = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, null);

        if (device_count == 0) {
            return error.NoGPUsSupportVulkan;
        }

        const devices = try self.allocator.alloc(vk.PhysicalDevice, device_count);
        defer self.allocator.free(devices);
        _ = try self.vki.enumeratePhysicalDevices(self.instance, &device_count, devices.ptr);

        for (devices) |device| {
            if (try self.isDeviceSuitable(device)) {
                self.physical_device = device;
                break;
            }
        }

        if (self.physical_device == .null_handle) {
            return error.NoSuitableDevice;
        }
    }

    fn createLogicalDevice(self: *Self) !void {
        const indices = try self.findQueueFamilies(self.physical_device);
        const queue_priority = [_]f32{1};

        var queue_create_info = [_]vk.DeviceQueueCreateInfo{
            .{
                .flags = .{},
                .queue_family_index = indices.graphics_family.?,
                .queue_count = 1,
                .p_queue_priorities = &queue_priority,
            },
            .{
                .flags = .{},
                .queue_family_index = indices.present_family.?,
                .queue_count = 1,
                .p_queue_priorities = &queue_priority,
            },
        };

        var create_info = vk.DeviceCreateInfo{
            .flags = .{},
            .queue_create_info_count = queue_create_info.len,
            .p_queue_create_infos = &queue_create_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = device_extensions.len,
            .pp_enabled_extension_names = &device_extensions,
            .p_enabled_features = null,
        };

        if (enable_validation_layers) {
            create_info.enabled_layer_count = validation_layers.len;
            create_info.pp_enabled_layer_names = &validation_layers;
        }

        self.device = try self.vki.createDevice(self.physical_device, &create_info, null);

        self.vkd = try DeviceDispatch.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr);

        self.graphics_queue = self.vkd.getDeviceQueue(self.device, indices.graphics_family.?, 0);
        self.present_queue = self.vkd.getDeviceQueue(self.device, indices.present_family.?, 0);
    }

    fn createSwapChain(self: *Self) !void {
        const swap_chain_support = try self.querySwapChainSupport(self.physical_device);
        defer swap_chain_support.deinit();

        const surface_format: vk.SurfaceFormatKHR = chooseSwapSurfaceFormat(swap_chain_support.formats.?);
        const present_mode: vk.PresentModeKHR = chooseSwapPresentMode(swap_chain_support.present_modes.?);
        const extent: vk.Extent2D = try self.chooseSwapExtent(swap_chain_support.capabilities);

        var image_count = swap_chain_support.capabilities.min_image_count + 1;
        if (swap_chain_support.capabilities.max_image_count > 0) {
            image_count = std.math.min(image_count, swap_chain_support.capabilities.max_image_count);
        }

        const indices = try self.findQueueFamilies(self.physical_device);
        const queue_family_indices = [_]u32{ indices.graphics_family.?, indices.present_family.? };
        const sharing_mode: vk.SharingMode = if (indices.graphics_family.? != indices.present_family.?)
            .concurrent
        else
            .exclusive;

        self.swap_chain = try self.vkd.createSwapchainKHR(self.device, &.{
            .flags = .{},
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = sharing_mode,
            .queue_family_index_count = queue_family_indices.len,
            .p_queue_family_indices = &queue_family_indices,
            .pre_transform = swap_chain_support.capabilities.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = .null_handle,
        }, null);

        _ = try self.vkd.getSwapchainImagesKHR(self.device, self.swap_chain, &image_count, null);
        self.swap_chain_images = try self.allocator.alloc(vk.Image, image_count);
        _ = try self.vkd.getSwapchainImagesKHR(self.device, self.swap_chain, &image_count, self.swap_chain_images.?.ptr);

        self.swap_chain_image_format = surface_format.format;
        self.swap_chain_extent = extent;
    }

    fn createImageViews(self: *Self) !void {
        self.swap_chain_image_views = try self.allocator.alloc(vk.ImageView, self.swap_chain_images.?.len);

        for (self.swap_chain_images.?) |image, i| {
            self.swap_chain_image_views.?[i] = try self.vkd.createImageView(self.device, &.{
                .flags = .{},
                .image = image,
                .view_type = .@"2d",
                .format = self.swap_chain_image_format,
                .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            }, null);
        }
    }

    fn createRenderPass(self: *Self) !void {
        const color_attachment = [_]vk.AttachmentDescription{.{
            .flags = .{},
            .format = self.swap_chain_image_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .present_src_khr,
        }};

        const color_attachment_ref = [_]vk.AttachmentReference{.{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        }};

        const subpass = [_]vk.SubpassDescription{.{
            .flags = .{},
            .pipeline_bind_point = .graphics,
            .input_attachment_count = 0,
            .p_input_attachments = undefined,
            .color_attachment_count = color_attachment_ref.len,
            .p_color_attachments = &color_attachment_ref,
            .p_resolve_attachments = null,
            .p_depth_stencil_attachment = null,
            .preserve_attachment_count = 0,
            .p_preserve_attachments = undefined,
        }};

        const dependencies = [_]vk.SubpassDependency{.{
            .src_subpass = vk.SUBPASS_EXTERNAL,
            .dst_subpass = 0,
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{},
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
            .dependency_flags = .{},
        }};

        self.render_pass = try self.vkd.createRenderPass(self.device, &.{
            .flags = .{},
            .attachment_count = color_attachment.len,
            .p_attachments = &color_attachment,
            .subpass_count = subpass.len,
            .p_subpasses = &subpass,
            .dependency_count = dependencies.len,
            .p_dependencies = &dependencies,
        }, null);
    }

    fn createGraphicsPipeline(self: *Self) !void {
        const vert_shader_module: vk.ShaderModule = try self.createShaderModule(resources.vert_18);
        defer self.vkd.destroyShaderModule(self.device, vert_shader_module, null);
        const frag_shader_module: vk.ShaderModule = try self.createShaderModule(resources.frag_18);
        defer self.vkd.destroyShaderModule(self.device, frag_shader_module, null);

        const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .flags = .{},
                .stage = .{ .vertex_bit = true },
                .module = vert_shader_module,
                .p_name = "main",
                .p_specialization_info = null,
            },
            .{
                .flags = .{},
                .stage = .{ .fragment_bit = true },
                .module = frag_shader_module,
                .p_name = "main",
                .p_specialization_info = null,
            },
        };

        const binding_description = Vertex.getBindingDescription();
        const attribute_descriptions = Vertex.getAttributeDescriptions();

        const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
            .flags = .{},
            .vertex_binding_description_count = 1,
            .p_vertex_binding_descriptions = @ptrCast([*]const vk.VertexInputBindingDescription, &binding_description),
            .vertex_attribute_description_count = attribute_descriptions.len,
            .p_vertex_attribute_descriptions = &attribute_descriptions,
        };

        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .flags = .{},
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };

        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .flags = .{},
            .viewport_count = 1,
            .p_viewports = undefined,
            .scissor_count = 1,
            .p_scissors = undefined,
        };

        const rasterizer = vk.PipelineRasterizationStateCreateInfo{
            .flags = .{},
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .flags = .{},
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = vk.FALSE,
            .min_sample_shading = 1,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const color_blend_attachment = [_]vk.PipelineColorBlendAttachmentState{.{
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        }};

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .flags = .{},
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = color_blend_attachment.len,
            .p_attachments = &color_blend_attachment,
            .blend_constants = [_]f32{ 0, 0, 0, 0 },
        };
        const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };

        const dynamic_state = vk.PipelineDynamicStateCreateInfo{
            .flags = .{},
            .dynamic_state_count = dynamic_states.len,
            .p_dynamic_states = &dynamic_states,
        };

        self.pipeline_layout = try self.vkd.createPipelineLayout(self.device, &.{
            .flags = .{},
            .set_layout_count = 0,
            .p_set_layouts = undefined,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);

        const pipeline_info = [_]vk.GraphicsPipelineCreateInfo{.{
            .flags = .{},
            .stage_count = shader_stages.len,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_tessellation_state = null,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = self.pipeline_layout,
            .render_pass = self.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        }};

        _ = try self.vkd.createGraphicsPipelines(
            self.device,
            .null_handle,
            pipeline_info.len,
            &pipeline_info,
            null,
            @ptrCast([*]vk.Pipeline, &self.graphics_pipeline),
        );
    }

    fn createFramebuffers(self: *Self) !void {
        self.swap_chain_framebuffers = try self.allocator.alloc(vk.Framebuffer, self.swap_chain_image_views.?.len);

        for (self.swap_chain_framebuffers.?) |*framebuffer, i| {
            const attachments = [_]vk.ImageView{self.swap_chain_image_views.?[i]};

            framebuffer.* = try self.vkd.createFramebuffer(self.device, &.{
                .flags = .{},
                .render_pass = self.render_pass,
                .attachment_count = attachments.len,
                .p_attachments = &attachments,
                .width = self.swap_chain_extent.width,
                .height = self.swap_chain_extent.height,
                .layers = 1,
            }, null);
        }
    }

    fn createCommandPool(self: *Self) !void {
        const queue_family_indices = try self.findQueueFamilies(self.physical_device);

        self.command_pool = try self.vkd.createCommandPool(self.device, &.{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_indices.graphics_family.?,
        }, null);
    }

    fn createVertexBuffer(self: *Self) !void {
        const buffer_size: vk.DeviceSize = @sizeOf(@TypeOf(vertices));

        var staging_buffer: vk.Buffer = undefined;
        var staging_buffer_memory: vk.DeviceMemory = undefined;
        try createBuffer(self, buffer_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, &staging_buffer, &staging_buffer_memory);

        const data = try self.vkd.mapMemory(self.device, staging_buffer_memory, 0, buffer_size, .{});
        // copy vertices to data
        const aligned_data = @ptrCast([*]Vertex, @alignCast(@alignOf(Vertex), data));
        for (vertices) |vertex, i| {
            aligned_data[i] = vertex;
        }
        self.vkd.unmapMemory(self.device, staging_buffer_memory);

        try createBuffer(self, buffer_size, .{ .transfer_dst_bit = true, .vertex_buffer_bit = true }, .{ .device_local_bit = true }, &self.vertex_buffer, &self.vertex_buffer_memory);

        try copyBuffer(self, staging_buffer, self.vertex_buffer, buffer_size);

        self.vkd.destroyBuffer(self.device, staging_buffer, null);
        self.vkd.freeMemory(self.device, staging_buffer_memory, null);
    }

    fn createBuffer(self: *Self, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, buffer: *vk.Buffer, buffer_memory: *vk.DeviceMemory) !void {
        buffer.* = try self.vkd.createBuffer(self.device, &.{
            .flags = .{},
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        }, null);

        const mem_requirements = self.vkd.getBufferMemoryRequirements(self.device, buffer.*);

        buffer_memory.* = try self.vkd.allocateMemory(self.device, &.{
            .allocation_size = mem_requirements.size,
            .memory_type_index = try self.findMemoryType(mem_requirements.memory_type_bits, properties),
        }, null);
        try self.vkd.bindBufferMemory(self.device, buffer.*, buffer_memory.*, 0);
    }

    fn copyBuffer(self: *Self, src_buffer: vk.Buffer, dst_buffer: vk.Buffer, size: vk.DeviceSize) !void {
        const alloc_info = vk.CommandBufferAllocateInfo{
            .level = .primary,
            .command_pool = self.command_pool,
            .command_buffer_count = 1,
        };

        var command_buffer: vk.CommandBuffer = undefined;
        try self.vkd.allocateCommandBuffers(self.device, &alloc_info, @ptrCast([*]vk.CommandBuffer, &command_buffer));

        const begin_info = vk.CommandBufferBeginInfo{
            .flags = .{ .one_time_submit_bit = true },
            .p_inheritance_info = null,
        };

        try self.vkd.beginCommandBuffer(command_buffer, &begin_info);

        const copy_region = [_]vk.BufferCopy{.{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        }};
        self.vkd.cmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

        try self.vkd.endCommandBuffer(command_buffer);

        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = undefined,
            .p_wait_dst_stage_mask = undefined,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &command_buffer),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = undefined,
        };
        try self.vkd.queueSubmit(self.graphics_queue, 1, @ptrCast([*]const vk.SubmitInfo, &submit_info), .null_handle);
        try self.vkd.queueWaitIdle(self.graphics_queue);

        self.vkd.freeCommandBuffers(self.device, self.command_pool, 1, @ptrCast([*]const vk.CommandBuffer, &command_buffer));
    }

    fn findMemoryType(self: *Self, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        const mem_properties = self.vki.getPhysicalDeviceMemoryProperties(self.physical_device);
        for (mem_properties.memory_types[0..mem_properties.memory_type_count]) |mem_type, i| {
            if (type_filter & (@as(u32, 1) << @truncate(u5, i)) != 0 and mem_type.property_flags.contains(properties)) {
                return @truncate(u32, i);
            }
        }

        return error.NoSuitableMemoryType;
    }

    fn createCommandBuffers(self: *Self) !void {
        self.command_buffers = try self.allocator.alloc(vk.CommandBuffer, MAX_FRAMES_IN_FLIGHT);

        try self.vkd.allocateCommandBuffers(self.device, &.{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = @intCast(u32, self.command_buffers.?.len),
        }, self.command_buffers.?.ptr);
    }

    fn recordCommandBuffer(self: *Self, command_buffer: vk.CommandBuffer, image_index: u32) !void {
        try self.vkd.beginCommandBuffer(command_buffer, &.{
            .flags = .{},
            .p_inheritance_info = null,
        });

        const clear_values = [_]vk.ClearValue{.{
            .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
        }};

        const render_pass_info = vk.RenderPassBeginInfo{
            .render_pass = self.render_pass,
            .framebuffer = self.swap_chain_framebuffers.?[image_index],
            .render_area = vk.Rect2D{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swap_chain_extent,
            },
            .clear_value_count = clear_values.len,
            .p_clear_values = &clear_values,
        };

        self.vkd.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");
        {
            self.vkd.cmdBindPipeline(command_buffer, .graphics, self.graphics_pipeline);

            const viewports = [_]vk.Viewport{.{
                .x = 0,
                .y = 0,
                .width = @intToFloat(f32, self.swap_chain_extent.width),
                .height = @intToFloat(f32, self.swap_chain_extent.height),
                .min_depth = 0,
                .max_depth = 1,
            }};
            self.vkd.cmdSetViewport(command_buffer, 0, viewports.len, &viewports);

            const scissors = [_]vk.Rect2D{.{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swap_chain_extent,
            }};
            self.vkd.cmdSetScissor(command_buffer, 0, scissors.len, &scissors);

            const vertex_buffers = [_]vk.Buffer{self.vertex_buffer};
            const offsets = [_]vk.DeviceSize{0};
            self.vkd.cmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffers, &offsets);

            self.vkd.cmdDraw(command_buffer, vertices.len, 1, 0, 0);
        }
        self.vkd.cmdEndRenderPass(command_buffer);

        try self.vkd.endCommandBuffer(command_buffer);
    }

    fn createSyncObjects(self: *Self) !void {
        self.image_available_semaphores = try self.allocator.alloc(vk.Semaphore, MAX_FRAMES_IN_FLIGHT);
        self.render_finished_semaphores = try self.allocator.alloc(vk.Semaphore, MAX_FRAMES_IN_FLIGHT);
        self.in_flight_fences = try self.allocator.alloc(vk.Fence, MAX_FRAMES_IN_FLIGHT);

        var i: usize = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            self.image_available_semaphores.?[i] = try self.vkd.createSemaphore(self.device, &.{ .flags = .{} }, null);
            self.render_finished_semaphores.?[i] = try self.vkd.createSemaphore(self.device, &.{ .flags = .{} }, null);
            self.in_flight_fences.?[i] = try self.vkd.createFence(self.device, &.{ .flags = .{ .signaled_bit = true } }, null);
        }
    }

    fn drawFrame(self: *Self) !void {
        _ = try self.vkd.waitForFences(self.device, 1, @ptrCast([*]const vk.Fence, &self.in_flight_fences.?[self.current_frame]), vk.TRUE, std.math.maxInt(u64));

        const result = self.vkd.acquireNextImageKHR(self.device, self.swap_chain, std.math.maxInt(u64), self.image_available_semaphores.?[self.current_frame], .null_handle) catch |err| switch (err) {
            error.OutOfDateKHR => {
                try self.recreateSwapChain();
                return;
            },
            else => |e| return e,
        };

        if (result.result != .success and result.result != .suboptimal_khr) {
            return error.ImageAcquireFailed;
        }

        try self.vkd.resetFences(self.device, 1, @ptrCast([*]const vk.Fence, &self.in_flight_fences.?[self.current_frame]));

        try self.vkd.resetCommandBuffer(self.command_buffers.?[self.current_frame], .{});
        try self.recordCommandBuffer(self.command_buffers.?[self.current_frame], result.image_index);

        const wait_semaphores = [_]vk.Semaphore{self.image_available_semaphores.?[self.current_frame]};
        const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
        const signal_semaphores = [_]vk.Semaphore{self.render_finished_semaphores.?[self.current_frame]};

        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = wait_semaphores.len,
            .p_wait_semaphores = &wait_semaphores,
            .p_wait_dst_stage_mask = &wait_stages,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &self.command_buffers.?[self.current_frame]),
            .signal_semaphore_count = signal_semaphores.len,
            .p_signal_semaphores = &signal_semaphores,
        };
        _ = try self.vkd.queueSubmit(self.graphics_queue, 1, &[_]vk.SubmitInfo{submit_info}, self.in_flight_fences.?[self.current_frame]);

        var present_result = self.vkd.queuePresentKHR(self.present_queue, &.{
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

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn createShaderModule(self: *Self, code: []const u8) !vk.ShaderModule {
        return try self.vkd.createShaderModule(self.device, &.{
            .flags = .{},
            .code_size = code.len,
            .p_code = @ptrCast([*]const u32, @alignCast(@alignOf(u32), code)),
        }, null);
    }

    fn chooseSwapSurfaceFormat(available_formats: []vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
        for (available_formats) |available_format| {
            if (available_format.format == .b8g8r8a8_srgb and available_format.color_space == .srgb_nonlinear_khr) {
                return available_format;
            }
        }

        return available_formats[0];
    }

    fn chooseSwapPresentMode(available_present_modes: []vk.PresentModeKHR) vk.PresentModeKHR {
        for (available_present_modes) |available_present_mode| {
            if (available_present_mode == .mailbox_khr) {
                return available_present_mode;
            }
        }

        return .fifo_khr;
    }

    fn chooseSwapExtent(self: *Self, capabilities: vk.SurfaceCapabilitiesKHR) !vk.Extent2D {
        if (capabilities.current_extent.width != 0xFFFF_FFFF) {
            return capabilities.current_extent;
        } else {
            const window_size = try self.window.?.getFramebufferSize();

            return vk.Extent2D{
                .width = std.math.clamp(window_size.width, capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                .height = std.math.clamp(window_size.height, capabilities.min_image_extent.height, capabilities.max_image_extent.height),
            };
        }
    }

    fn querySwapChainSupport(self: *Self, device: vk.PhysicalDevice) !SwapChainSupportDetails {
        var details = SwapChainSupportDetails.init(self.allocator);

        details.capabilities = try self.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(device, self.surface);

        var format_count: u32 = undefined;
        _ = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, null);

        if (format_count != 0) {
            details.formats = try details.allocator.alloc(vk.SurfaceFormatKHR, format_count);
            _ = try self.vki.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface, &format_count, details.formats.?.ptr);
        }

        var present_mode_count: u32 = undefined;
        _ = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, null);

        if (present_mode_count != 0) {
            details.present_modes = try details.allocator.alloc(vk.PresentModeKHR, present_mode_count);
            _ = try self.vki.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface, &present_mode_count, details.present_modes.?.ptr);
        }

        return details;
    }

    fn isDeviceSuitable(self: *Self, device: vk.PhysicalDevice) !bool {
        const indices = try self.findQueueFamilies(device);

        const extensions_supported = try self.checkDeviceExtensionSupport(device);

        var swap_chain_adequate = false;
        if (extensions_supported) {
            const swap_chain_support = try self.querySwapChainSupport(device);
            defer swap_chain_support.deinit();

            swap_chain_adequate = swap_chain_support.formats != null and swap_chain_support.present_modes != null;
        }

        return indices.isComplete() and extensions_supported and swap_chain_adequate;
    }

    fn checkDeviceExtensionSupport(self: *Self, device: vk.PhysicalDevice) !bool {
        var extension_count: u32 = undefined;
        _ = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, null);

        const available_extensions = try self.allocator.alloc(vk.ExtensionProperties, extension_count);
        defer self.allocator.free(available_extensions);
        _ = try self.vki.enumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);

        const required_extensions = device_extensions[0..];

        for (required_extensions) |required_extension| {
            for (available_extensions) |available_extension| {
                const len = std.mem.indexOfScalar(u8, &available_extension.extension_name, 0).?;
                const available_extension_name = available_extension.extension_name[0..len];
                if (std.mem.eql(u8, std.mem.span(required_extension), available_extension_name)) {
                    break;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    fn findQueueFamilies(self: *Self, device: vk.PhysicalDevice) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{};

        var queue_family_count: u32 = 0;
        self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

        const queue_families = try self.allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
        defer self.allocator.free(queue_families);
        self.vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

        for (queue_families) |queue_family, i| {
            if (indices.graphics_family == null and queue_family.queue_flags.graphics_bit) {
                indices.graphics_family = @intCast(u32, i);
            } else if (indices.present_family == null and (try self.vki.getPhysicalDeviceSurfaceSupportKHR(device, @intCast(u32, i), self.surface)) == vk.TRUE) {
                indices.present_family = @intCast(u32, i);
            }

            if (indices.isComplete()) {
                break;
            }
        }

        return indices;
    }

    fn getRequiredExtensions(allocator: Allocator) !std.ArrayListAligned([*:0]const u8, null) {
        var extensions = std.ArrayList([*:0]const u8).init(allocator);
        try extensions.appendSlice(try glfw.getRequiredInstanceExtensions());

        if (enable_validation_layers) {
            try extensions.append(vk.extension_info.ext_debug_utils.name);
        }

        return extensions;
    }

    fn checkValidationLayerSupport(self: *Self) !bool {
        var layer_count: u32 = undefined;
        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);

        var available_layers = try self.allocator.alloc(vk.LayerProperties, layer_count);
        defer self.allocator.free(available_layers);
        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.ptr);

        for (validation_layers) |layer_name| {
            var layer_found: bool = false;

            for (available_layers) |layer_properties| {
                const available_len = std.mem.indexOfScalar(u8, &layer_properties.layer_name, 0).?;
                const available_layer_name = layer_properties.layer_name[0..available_len];
                if (std.mem.eql(u8, std.mem.span(layer_name), available_layer_name)) {
                    layer_found = true;
                    break;
                }
            }

            if (!layer_found) {
                return false;
            }
        }

        return true;
    }

    fn debugCallback(_: vk.DebugUtilsMessageSeverityFlagsEXT.IntType, _: vk.DebugUtilsMessageTypeFlagsEXT.IntType, p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT, _: ?*anyopaque) callconv(vk.vulkan_call_conv) vk.Bool32 {
        if (p_callback_data != null) {
            std.log.debug("validation layer: {s}", .{p_callback_data.?.p_message});
        }

        return vk.FALSE;
    }
};

pub fn main() void {
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
}
