const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const glfw = @import("glfw");
const vk = @import("vulkan");

// Vulkan pointer handles are non-optional pointers in current vulkan-zig.
// Use a non-null sentinel to represent "not created yet".
const null_instance: vk.Instance = @ptrFromInt(1);
const null_physical_device: vk.PhysicalDevice = @ptrFromInt(1);
const null_device: vk.Device = @ptrFromInt(1);
const null_queue: vk.Queue = @ptrFromInt(1);
const null_command_buffer: vk.CommandBuffer = @ptrFromInt(1);
const za = @import("zalgebra");
const resources = @import("resources");

const c = @import("stb");

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

const device_extensions = [_][*:0]const u8{vk.device_extensions.khr_swapchain.name};

const enable_validation_layers: bool = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    else => false,
};

const BaseDispatch = vk.BaseWrapper;

const InstanceDispatch = vk.InstanceWrapper;

const DeviceDispatch = vk.DeviceWrapper;

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

const UniformBufferObject = struct {
    model: za.Mat4 align(16),
    view: za.Mat4 align(16),
    proj: za.Mat4 align(16),
};

const vertices = [_]Vertex{
    .{ .pos = .{ -0.5, -0.5 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 0.5, -0.5 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ 0.5, 0.5 }, .color = .{ 0, 0, 1 } },
    .{ .pos = .{ -0.5, 0.5 }, .color = .{ 1, 1, 1 } },
};

const indices_input = [_]u16{ 0, 1, 2, 2, 3, 0 };

const HelloTriangleApplication = struct {
    const Self = @This();
    allocator: Allocator,

    window: ?*glfw.Window = null,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,
    vkd: DeviceDispatch = undefined,

    instance: vk.Instance = null_instance,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null,
    surface: vk.SurfaceKHR = .null,

    physical_device: vk.PhysicalDevice = null_physical_device,
    device: vk.Device = null_device,

    graphics_queue: vk.Queue = null_queue,
    present_queue: vk.Queue = null_queue,

    swap_chain: vk.SwapchainKHR = .null,
    swap_chain_images: ?[]vk.Image = null,
    swap_chain_image_format: vk.Format = .@"undefined",
    swap_chain_extent: vk.Extent2D = .{ .width = 0, .height = 0 },
    swap_chain_image_views: ?[]vk.ImageView = null,
    swap_chain_framebuffers: ?[]vk.Framebuffer = null,

    render_pass: vk.RenderPass = .null,
    descriptor_set_layout: vk.DescriptorSetLayout = .null,
    pipeline_layout: vk.PipelineLayout = .null,
    graphics_pipeline: vk.Pipeline = .null,

    command_pool: vk.CommandPool = .null,

    texture_image: vk.Image = .null,
    texture_image_memory: vk.DeviceMemory = .null,
    texture_image_view: vk.ImageView = .null,
    texture_sampler: vk.Sampler = .null,

    vertex_buffer: vk.Buffer = .null,
    vertex_buffer_memory: vk.DeviceMemory = .null,
    index_buffer: vk.Buffer = .null,
    index_buffer_memory: vk.DeviceMemory = .null,

    uniform_buffers: ?[]vk.Buffer = null,
    uniform_buffers_memory: ?[]vk.DeviceMemory = null,

    descriptor_pool: vk.DescriptorPool = .null,
    descriptor_sets: ?[]vk.DescriptorSet = null,

    command_buffers: ?[]vk.CommandBuffer = null,

    image_available_semaphores: ?[]vk.Semaphore = null,
    render_finished_semaphores: ?[]vk.Semaphore = null,
    in_flight_fences: ?[]vk.Fence = null,
    current_frame: u32 = 0,

    framebuffer_resized: bool = false,

    start_time: u64,

    pub fn init(allocator: Allocator) !Self {
        return Self{ .allocator = allocator, .start_time = nowNs() };
    }

    pub fn run(self: *Self) !void {
        try self.initWindow();
        try self.initVulkan();
        try self.mainLoop();
    }

    fn initWindow(self: *Self) !void {
        try glfw.init();
        glfw.windowHint(.client_api, .no_api);
        self.window = try glfw.Window.create(WIDTH, HEIGHT, "Vulkan", null, null);
        self.window.?.setUserPointer(self);
        _ = self.window.?.setFramebufferSizeCallback(framebufferResizeCallback);
    }

    fn framebufferResizeCallback(window: *glfw.Window, _: c_int, _: c_int) callconv(.c) void {
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
        try self.createDescriptorSetLayout();
        try self.createGraphicsPipeline();
        try self.createFramebuffers();
        try self.createCommandPool();
        try self.createTextureImage();
        try self.createTextureImageView();
        try self.createTextureSampler();
        try self.createVertexBuffer();
        try self.createIndexBuffer();
        try self.createUniformBuffers();
        try self.createDescriptorPool();
        try self.createDescriptorSets();
        try self.createCommandBuffers();
        try self.createSyncObjects();
    }

    fn mainLoop(self: *Self) !void {
        while (!self.window.?.shouldClose()) {
            glfw.pollEvents();
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

        if (self.swap_chain != .null) {
            self.vkd.destroySwapchainKHR(self.device, self.swap_chain, null);
            self.swap_chain = .null;
        }
    }

    pub fn deinit(self: *Self) void {
        self.cleanupSwapChain();

        if (self.graphics_pipeline != .null) self.vkd.destroyPipeline(self.device, self.graphics_pipeline, null);
        if (self.pipeline_layout != .null) self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
        if (self.render_pass != .null) self.vkd.destroyRenderPass(self.device, self.render_pass, null);

        if (self.uniform_buffers != null) {
            for (self.uniform_buffers.?) |uniform_buffer| {
                self.vkd.destroyBuffer(self.device, uniform_buffer, null);
            }
            self.allocator.free(self.uniform_buffers.?);
        }
        if (self.uniform_buffers_memory != null) {
            for (self.uniform_buffers_memory.?) |uniform_buffer_memory| {
                self.vkd.freeMemory(self.device, uniform_buffer_memory, null);
            }
            self.allocator.free(self.uniform_buffers_memory.?);
        }

        if (self.descriptor_pool != .null) self.vkd.destroyDescriptorPool(self.device, self.descriptor_pool, null);
        if (self.descriptor_sets != null) self.allocator.free(self.descriptor_sets.?);

        if (self.texture_sampler != .null) self.vkd.destroySampler(self.device, self.texture_sampler, null);
        if (self.texture_image_view != .null) self.vkd.destroyImageView(self.device, self.texture_image_view, null);

        if (self.texture_image != .null) self.vkd.destroyImage(self.device, self.texture_image, null);
        if (self.texture_image_memory != .null) self.vkd.freeMemory(self.device, self.texture_image_memory, null);

        if (self.descriptor_set_layout != .null) self.vkd.destroyDescriptorSetLayout(self.device, self.descriptor_set_layout, null);

        if (self.index_buffer != .null) self.vkd.destroyBuffer(self.device, self.index_buffer, null);
        if (self.index_buffer_memory != .null) self.vkd.freeMemory(self.device, self.index_buffer_memory, null);

        if (self.vertex_buffer != .null) self.vkd.destroyBuffer(self.device, self.vertex_buffer, null);
        if (self.vertex_buffer_memory != .null) self.vkd.freeMemory(self.device, self.vertex_buffer_memory, null);

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

        if (self.command_pool != .null) self.vkd.destroyCommandPool(self.device, self.command_pool, null);
        if (self.command_buffers != null) self.allocator.free(self.command_buffers.?);

        if (self.device != null_device) self.vkd.destroyDevice(self.device, null);

        if (enable_validation_layers and self.debug_messenger != .null) self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);

        if (self.surface != .null) self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        if (self.instance != null_instance) self.vki.destroyInstance(self.instance, null);

        if (self.window != null) self.window.?.destroy();

        glfw.terminate();
    }

    fn recreateSwapChain(self: *Self) !void {
        var size = self.window.?.getFramebufferSize();

        while (size[0] == 0 or size[1] == 0) {
            size = self.window.?.getFramebufferSize();
            glfw.waitEvents();
        }

        try self.vkd.deviceWaitIdle(self.device);

        self.cleanupSwapChain();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createFramebuffers();
    }

    fn createInstance(self: *Self) !void {
        const vk_proc = @as(*const fn (instance: ?vk.Instance, procname: [*:0]const u8) callconv(.c) vk.PfnVoidFunction, @ptrCast(&glfw.getInstanceProcAddress));
        self.vkb = BaseDispatch.load(vk_proc);

        if (enable_validation_layers and !try self.checkValidationLayerSupport()) {
            return error.MissingValidationLayer;
        }

        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(1, 0, 0, 0).toU32(),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0, 0).toU32(),
            .api_version = vk.API_VERSION_1_2.toU32(),
        };

        const extensions = try getRequiredExtensions(self.allocator);
        defer extensions.deinit();

        var create_info = vk.InstanceCreateInfo{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = null,
            .enabled_extension_count = @as(u32, @intCast(extensions.items.len)),
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

        self.vki = InstanceDispatch.load(self.instance, vk_proc);
    }

    fn populateDebugMessengerCreateInfo(create_info: *vk.DebugUtilsMessengerCreateInfoEXT) void {
        create_info.* = .{
            .flags = .{},
            .message_severity = .{
                .verbose_ext = true,
                .warning_ext = true,
                .error_ext = true,
            },
            .message_type = .{
                .general_ext = true,
                .validation_ext = true,
                .performance_ext = true,
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
        try glfw.createWindowSurface(self.instance, self.window.?, null, &self.surface);
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

        if (self.physical_device == null_physical_device) {
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

        const device_features = vk.PhysicalDeviceFeatures{
            .sampler_anisotropy = .true,
        };

        var create_info = vk.DeviceCreateInfo{
            .flags = .{},
            .queue_create_info_count = queue_create_info.len,
            .p_queue_create_infos = &queue_create_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = null,
            .enabled_extension_count = device_extensions.len,
            .pp_enabled_extension_names = &device_extensions,
            .p_enabled_features = &device_features,
        };


        self.device = try self.vki.createDevice(self.physical_device, &create_info, null);

        self.vkd = DeviceDispatch.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr.?);

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
            image_count = @min(image_count, swap_chain_support.capabilities.max_image_count);
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
            .image_usage = .{ .color_attachment = true },
            .image_sharing_mode = sharing_mode,
            .queue_family_index_count = queue_family_indices.len,
            .p_queue_family_indices = &queue_family_indices,
            .pre_transform = swap_chain_support.capabilities.current_transform,
            .composite_alpha = .{ .opaque_khr = true },
            .present_mode = present_mode,
            .clipped = .true,
            .old_swapchain = .null,
        }, null);

        _ = try self.vkd.getSwapchainImagesKHR(self.device, self.swap_chain, &image_count, null);
        self.swap_chain_images = try self.allocator.alloc(vk.Image, image_count);
        _ = try self.vkd.getSwapchainImagesKHR(self.device, self.swap_chain, &image_count, self.swap_chain_images.?.ptr);

        self.swap_chain_image_format = surface_format.format;
        self.swap_chain_extent = extent;
    }

    fn createImageViews(self: *Self) !void {
        self.swap_chain_image_views = try self.allocator.alloc(vk.ImageView, self.swap_chain_images.?.len);

        for (self.swap_chain_images.?, 0..) |image, i| {
            self.swap_chain_image_views.?[i] = try self.createImageView(image, self.swap_chain_image_format);
        }
    }

    fn createRenderPass(self: *Self) !void {
        const color_attachment = [_]vk.AttachmentDescription{.{
            .flags = .{},
            .format = self.swap_chain_image_format,
            .samples = .{ .@"1" = true },
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
            .src_stage_mask = .{ .color_attachment_output = true },
            .src_access_mask = .{},
            .dst_stage_mask = .{ .color_attachment_output = true },
            .dst_access_mask = .{ .color_attachment_write = true },
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

    fn createDescriptorSetLayout(self: *Self) !void {
        const ubo_layout_binding = [_]vk.DescriptorSetLayoutBinding{.{
            .binding = 0,
            .descriptor_count = 1,
            .descriptor_type = .uniform_buffer,
            .p_immutable_samplers = null,
            .stage_flags = .{ .vertex = true },
        }};

        const layout_info = vk.DescriptorSetLayoutCreateInfo{
            .flags = .{},
            .binding_count = ubo_layout_binding.len,
            .p_bindings = &ubo_layout_binding,
        };

        self.descriptor_set_layout = try self.vkd.createDescriptorSetLayout(self.device, &layout_info, null);
    }

    fn createGraphicsPipeline(self: *Self) !void {
        const vert_shader_module: vk.ShaderModule = try self.createShaderModule(resources.vert_22);
        defer self.vkd.destroyShaderModule(self.device, vert_shader_module, null);
        const frag_shader_module: vk.ShaderModule = try self.createShaderModule(resources.frag_22);
        defer self.vkd.destroyShaderModule(self.device, frag_shader_module, null);

        const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .flags = .{},
                .stage = .{ .vertex = true },
                .module = vert_shader_module,
                .p_name = "main",
                .p_specialization_info = null,
            },
            .{
                .flags = .{},
                .stage = .{ .fragment = true },
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
            .p_vertex_binding_descriptions = @as([*]const vk.VertexInputBindingDescription, @ptrCast(&binding_description)),
            .vertex_attribute_description_count = attribute_descriptions.len,
            .p_vertex_attribute_descriptions = &attribute_descriptions,
        };

        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .flags = .{},
            .topology = .triangle_list,
            .primitive_restart_enable = .false,
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
            .depth_clamp_enable = .false,
            .rasterizer_discard_enable = .false,
            .polygon_mode = .fill,
            .cull_mode = .{ .back = true },
            .front_face = .counter_clockwise,
            .depth_bias_enable = .false,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .flags = .{},
            .rasterization_samples = .{ .@"1" = true },
            .sample_shading_enable = .false,
            .min_sample_shading = 1,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = .false,
            .alpha_to_one_enable = .false,
        };

        const color_blend_attachment = [_]vk.PipelineColorBlendAttachmentState{.{
            .blend_enable = .false,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r = true, .g = true, .b = true, .a = true },
        }};

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .flags = .{},
            .logic_op_enable = .false,
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
            .set_layout_count = 1,
            .p_set_layouts = @as([*]const vk.DescriptorSetLayout, @ptrCast(&self.descriptor_set_layout)),
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
            .base_pipeline_handle = .null,
            .base_pipeline_index = -1,
        }};

        _ = try self.vkd.createGraphicsPipelines(
            self.device,
            .null,
            &pipeline_info,
            null,
            (&self.graphics_pipeline)[0..1],
        );
    }

    fn createFramebuffers(self: *Self) !void {
        self.swap_chain_framebuffers = try self.allocator.alloc(vk.Framebuffer, self.swap_chain_image_views.?.len);

        for (self.swap_chain_framebuffers.?, 0..) |*framebuffer, i| {
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
            .flags = .{ .reset_command_buffer = true },
            .queue_family_index = queue_family_indices.graphics_family.?,
        }, null);
    }

    fn createTextureImage(self: *Self) !void {
        var tex_width: c_int = undefined;
        var tex_height: c_int = undefined;
        var channels: c_int = undefined;
        const pixels = c.stbi_load("resources/texture.jpg", &tex_width, &tex_height, &channels, c.STBI_rgb_alpha);
        defer c.stbi_image_free(pixels);
        if (pixels == null) {
            return error.ImageLoadFailure;
        }

        const image_size: vk.DeviceSize = @as(u64, @intCast(tex_width)) * @as(u64, @intCast(tex_height)) * 4;

        var staging_buffer: vk.Buffer = undefined;
        var staging_buffer_memory: vk.DeviceMemory = undefined;
        try self.createBuffer(image_size, .{ .transfer_src = true }, .{ .host_visible = true, .host_coherent = true }, &staging_buffer, &staging_buffer_memory);

        const data = try self.vkd.mapMemory(self.device, staging_buffer_memory, 0, image_size, .{});
        @memcpy(@as([*]u8, @ptrCast(data.?))[0..image_size], pixels[0..image_size]);
        self.vkd.unmapMemory(self.device, staging_buffer_memory);

        try self.createImage(@as(u32, @intCast(tex_width)), @as(u32, @intCast(tex_height)), .r8g8b8a8_srgb, .optimal, .{ .transfer_dst = true, .sampled = true }, .{ .device_local = true }, &self.texture_image, &self.texture_image_memory);

        try self.transitionImageLayout(self.texture_image, .r8g8b8a8_srgb, .@"undefined", .transfer_dst_optimal);
        try self.copyBufferToImage(staging_buffer, self.texture_image, @as(u32, @intCast(tex_width)), @as(u32, @intCast(tex_height)));
        try self.transitionImageLayout(self.texture_image, .r8g8b8a8_srgb, .transfer_dst_optimal, .shader_read_only_optimal);

        self.vkd.destroyBuffer(self.device, staging_buffer, null);
        self.vkd.freeMemory(self.device, staging_buffer_memory, null);
    }

    fn createTextureImageView(self: *Self) !void {
        self.texture_image_view = try self.createImageView(self.texture_image, .r8g8b8a8_srgb);
    }

    fn createTextureSampler(self: *Self) !void {
        const properties = self.vki.getPhysicalDeviceProperties(self.physical_device);

        const sampler_info = vk.SamplerCreateInfo{
            .flags = .{},
            .mip_lod_bias = 0,
            .min_lod = 0,
            .max_lod = 0,
            .mag_filter = .linear,
            .min_filter = .linear,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .anisotropy_enable = .true,
            .max_anisotropy = properties.limits.max_sampler_anisotropy,
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = .false,
            .compare_enable = .false,
            .compare_op = .always,
            .mipmap_mode = .linear,
        };

        self.texture_sampler = try self.vkd.createSampler(self.device, &sampler_info, null);
    }

    fn createImageView(self: *Self, image: vk.Image, format: vk.Format) !vk.ImageView {
        const view_info = vk.ImageViewCreateInfo{
            .flags = .{},
            .image = image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };
        return try self.vkd.createImageView(self.device, &view_info, null);
    }

    fn createImage(self: *Self, image_width: u32, image_height: u32, format: vk.Format, tiling: vk.ImageTiling, usage: vk.ImageUsageFlags, properties: vk.MemoryPropertyFlags, image: *vk.Image, image_memory: *vk.DeviceMemory) !void {
        const image_info = vk.ImageCreateInfo{
            .flags = .{},
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
            .image_type = .@"2d",
            .extent = .{
                .width = image_width,
                .height = image_height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .format = format,
            .tiling = tiling,
            .initial_layout = .@"undefined",
            .usage = usage,
            .samples = .{ .@"1" = true },
            .sharing_mode = .exclusive,
        };

        image.* = try self.vkd.createImage(self.device, &image_info, null);

        const mem_requirements = self.vkd.getImageMemoryRequirements(self.device, image.*);

        const alloc_info = vk.MemoryAllocateInfo{
            .allocation_size = mem_requirements.size,
            .memory_type_index = try self.findMemoryType(mem_requirements.memory_type_bits, properties),
        };
        image_memory.* = try self.vkd.allocateMemory(self.device, &alloc_info, null);
        try self.vkd.bindImageMemory(self.device, image.*, image_memory.*, 0);
    }

    fn transitionImageLayout(self: *Self, image: vk.Image, _: vk.Format, old_layout: vk.ImageLayout, new_layout: vk.ImageLayout) !void {
        const command_buffer = try self.beginSingleTimeCommands();

        var barrier = [_]vk.ImageMemoryBarrier{.{
            .old_layout = old_layout,
            .new_layout = new_layout,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .src_access_mask = .{},
            .dst_access_mask = .{},
            .image = image,
            .subresource_range = .{
                .aspect_mask = .{ .color = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }};

        var source_stage: vk.PipelineStageFlags = undefined;
        var destination_stage: vk.PipelineStageFlags = undefined;

        if (old_layout == .@"undefined" and new_layout == .transfer_dst_optimal) {
            barrier[0].src_access_mask = .{};
            barrier[0].dst_access_mask = .{ .transfer_write = true };

            source_stage = .{ .top_of_pipe = true };
            destination_stage = .{ .transfer = true };
        } else if (old_layout == .transfer_dst_optimal and new_layout == .shader_read_only_optimal) {
            barrier[0].src_access_mask = .{ .transfer_write = true };
            barrier[0].dst_access_mask = .{ .shader_read = true };

            source_stage = .{ .transfer = true };
            destination_stage = .{ .fragment_shader = true };
        } else {
            return error.UnsupportedLayoutTransition;
        }

        self.vkd.cmdPipelineBarrier(command_buffer, source_stage, destination_stage, .{}, null, null, &barrier);

        try self.endSingleTimeCommands(command_buffer);
    }

    fn copyBufferToImage(self: *Self, buffer: vk.Buffer, image: vk.Image, image_width: u32, image_height: u32) !void {
        const command_buffer = try self.beginSingleTimeCommands();

        const region = [_]vk.BufferImageCopy{.{
            .buffer_offset = 0,
            .buffer_row_length = 0,
            .buffer_image_height = 0,
            .image_subresource = .{
                .aspect_mask = .{ .color = true },
                .mip_level = 0,
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            .image_extent = .{ .width = image_width, .height = image_height, .depth = 1 },
        }};

        self.vkd.cmdCopyBufferToImage(command_buffer, buffer, image, .transfer_dst_optimal, &region);

        try self.endSingleTimeCommands(command_buffer);
    }

    fn createVertexBuffer(self: *Self) !void {
        const buffer_size: vk.DeviceSize = @sizeOf(@TypeOf(vertices));

        var staging_buffer: vk.Buffer = undefined;
        var staging_buffer_memory: vk.DeviceMemory = undefined;
        try createBuffer(self, buffer_size, .{ .transfer_src = true }, .{ .host_visible = true, .host_coherent = true }, &staging_buffer, &staging_buffer_memory);

        const data = try self.vkd.mapMemory(self.device, staging_buffer_memory, 0, buffer_size, .{});
        @memcpy(@as([*]u8, @ptrCast(data.?))[0..buffer_size], std.mem.sliceAsBytes(&vertices));
        self.vkd.unmapMemory(self.device, staging_buffer_memory);

        try createBuffer(self, buffer_size, .{ .transfer_dst = true, .vertex_buffer = true }, .{ .device_local = true }, &self.vertex_buffer, &self.vertex_buffer_memory);

        try copyBuffer(self, staging_buffer, self.vertex_buffer, buffer_size);

        self.vkd.destroyBuffer(self.device, staging_buffer, null);
        self.vkd.freeMemory(self.device, staging_buffer_memory, null);
    }

    fn createIndexBuffer(self: *Self) !void {
        const buffer_size: vk.DeviceSize = @sizeOf(@TypeOf(indices_input));

        var staging_buffer: vk.Buffer = undefined;
        var staging_buffer_memory: vk.DeviceMemory = undefined;
        try createBuffer(self, buffer_size, .{ .transfer_src = true }, .{ .host_visible = true, .host_coherent = true }, &staging_buffer, &staging_buffer_memory);

        const data = try self.vkd.mapMemory(self.device, staging_buffer_memory, 0, buffer_size, .{});
        @memcpy(@as([*]u8, @ptrCast(data.?))[0..buffer_size], std.mem.sliceAsBytes(&indices_input));
        self.vkd.unmapMemory(self.device, staging_buffer_memory);

        try createBuffer(self, buffer_size, .{ .transfer_dst = true, .index_buffer = true }, .{ .device_local = true }, &self.index_buffer, &self.index_buffer_memory);

        try copyBuffer(self, staging_buffer, self.index_buffer, buffer_size);

        self.vkd.destroyBuffer(self.device, staging_buffer, null);
        self.vkd.freeMemory(self.device, staging_buffer_memory, null);
    }

    fn createUniformBuffers(self: *Self) !void {
        const buffer_size: vk.DeviceSize = @sizeOf(UniformBufferObject);

        self.uniform_buffers = try self.allocator.alloc(vk.Buffer, MAX_FRAMES_IN_FLIGHT);
        self.uniform_buffers_memory = try self.allocator.alloc(vk.DeviceMemory, MAX_FRAMES_IN_FLIGHT);

        var i: usize = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            try self.createBuffer(buffer_size, .{ .uniform_buffer = true }, .{ .host_visible = true, .host_coherent = true }, &self.uniform_buffers.?[i], &self.uniform_buffers_memory.?[i]);
        }
    }

    fn createDescriptorPool(self: *Self) !void {
        const pool_sizes = [_]vk.DescriptorPoolSize{.{
            .type = .uniform_buffer,
            .descriptor_count = MAX_FRAMES_IN_FLIGHT,
        }};
        const pool_info = vk.DescriptorPoolCreateInfo{
            .flags = .{},
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
            .max_sets = MAX_FRAMES_IN_FLIGHT,
        };
        self.descriptor_pool = try self.vkd.createDescriptorPool(self.device, &pool_info, null);
    }

    fn createDescriptorSets(self: *Self) !void {
        const layouts = [_]vk.DescriptorSetLayout{ self.descriptor_set_layout, self.descriptor_set_layout };
        const alloc_info = vk.DescriptorSetAllocateInfo{
            .descriptor_pool = self.descriptor_pool,
            .descriptor_set_count = MAX_FRAMES_IN_FLIGHT,
            .p_set_layouts = &layouts,
        };
        self.descriptor_sets = try self.allocator.alloc(vk.DescriptorSet, MAX_FRAMES_IN_FLIGHT);

        try self.vkd.allocateDescriptorSets(self.device, &alloc_info, self.descriptor_sets.?.ptr);

        for (self.descriptor_sets.?, 0..) |descriptor_set, i| {
            const buffer_info = [_]vk.DescriptorBufferInfo{.{
                .buffer = self.uniform_buffers.?[i],
                .offset = 0,
                .range = @sizeOf(UniformBufferObject),
            }};

            const descriptor_write = [_]vk.WriteDescriptorSet{.{
                .dst_set = descriptor_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
                .p_buffer_info = &buffer_info,
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            }};

            self.vkd.updateDescriptorSets(self.device, &descriptor_write, null);
        }
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

    fn beginSingleTimeCommands(self: *Self) !vk.CommandBuffer {
        const alloc_info = vk.CommandBufferAllocateInfo{
            .level = .primary,
            .command_pool = self.command_pool,
            .command_buffer_count = 1,
        };

        var command_buffer: vk.CommandBuffer = undefined;
        try self.vkd.allocateCommandBuffers(self.device, &alloc_info, @as([*]vk.CommandBuffer, @ptrCast(&command_buffer)));

        const begin_info = vk.CommandBufferBeginInfo{
            .flags = .{ .one_time_submit = true },
            .p_inheritance_info = null,
        };

        try self.vkd.beginCommandBuffer(command_buffer, &begin_info);

        return command_buffer;
    }

    fn endSingleTimeCommands(self: *Self, command_buffer: vk.CommandBuffer) !void {
        try self.vkd.endCommandBuffer(command_buffer);

        const submit_infos = [_]vk.SubmitInfo{.{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = undefined,
            .p_wait_dst_stage_mask = undefined,
            .command_buffer_count = 1,
            .p_command_buffers = @as([*]const vk.CommandBuffer, @ptrCast(&command_buffer)),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = undefined,
        }};
        try self.vkd.queueSubmit(self.graphics_queue, &submit_infos, .null);
        try self.vkd.queueWaitIdle(self.graphics_queue);

        self.vkd.freeCommandBuffers(self.device, self.command_pool, (&command_buffer)[0..1]);
    }

    fn copyBuffer(self: *Self, src_buffer: vk.Buffer, dst_buffer: vk.Buffer, size: vk.DeviceSize) !void {
        const command_buffer = try self.beginSingleTimeCommands();

        const copy_region = [_]vk.BufferCopy{.{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        }};
        self.vkd.cmdCopyBuffer(command_buffer, src_buffer, dst_buffer, &copy_region);

        try self.endSingleTimeCommands(command_buffer);
    }

    fn findMemoryType(self: *Self, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        const mem_properties = self.vki.getPhysicalDeviceMemoryProperties(self.physical_device);
        for (mem_properties.memory_types[0..mem_properties.memory_type_count], 0..) |mem_type, i| {
            if (type_filter & (@as(u32, 1) << @as(u5, @truncate(i))) != 0 and mem_type.property_flags.contains(properties)) {
                return @as(u32, @truncate(i));
            }
        }

        return error.NoSuitableMemoryType;
    }

    fn createCommandBuffers(self: *Self) !void {
        self.command_buffers = try self.allocator.alloc(vk.CommandBuffer, MAX_FRAMES_IN_FLIGHT);

        try self.vkd.allocateCommandBuffers(self.device, &.{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = @as(u32, @intCast(self.command_buffers.?.len)),
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
                .width = @as(f32, @floatFromInt(self.swap_chain_extent.width)),
                .height = @as(f32, @floatFromInt(self.swap_chain_extent.height)),
                .min_depth = 0,
                .max_depth = 1,
            }};
            self.vkd.cmdSetViewport(command_buffer, 0, &viewports);

            const scissors = [_]vk.Rect2D{.{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swap_chain_extent,
            }};
            self.vkd.cmdSetScissor(command_buffer, 0, &scissors);

            const vertex_buffers = [_]vk.Buffer{self.vertex_buffer};
            const offsets = [_]vk.DeviceSize{0};
            self.vkd.cmdBindVertexBuffers(command_buffer, 0, &vertex_buffers, &offsets);

            self.vkd.cmdBindIndexBuffer(command_buffer, self.index_buffer, 0, vk.IndexType.uint16);

            self.vkd.cmdBindDescriptorSets(command_buffer, .graphics, self.pipeline_layout, 0, (&self.descriptor_sets.?[self.current_frame])[0..1], null);

            self.vkd.cmdDrawIndexed(command_buffer, indices_input.len, 1, 0, 0, 0);
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
            self.in_flight_fences.?[i] = try self.vkd.createFence(self.device, &.{ .flags = .{ .signaled = true } }, null);
        }
    }

    fn updateUniformBuffer(self: *Self, current_image: u32) !void {
        const time: f32 = (@as(f32, @floatFromInt(nowNs() - self.start_time)) / std.time.ns_per_s);

        var ubo = UniformBufferObject{
            .model = za.Mat4.identity().rotate(time * 90.0, za.Vec3.new(0.0, 0.0, 1.0)),
            .view = za.lookAt(za.Vec3.new(2, 2, 2), za.Vec3.new(0, 0, 0), za.Vec3.new(0, 0, 1)),
            .proj = za.perspective(45.0, @as(f32, @floatFromInt(self.swap_chain_extent.width)) / @as(f32, @floatFromInt(self.swap_chain_extent.height)), 0.1, 10),
        };
        ubo.proj.data[1][1] *= -1;

        const data = try self.vkd.mapMemory(self.device, self.uniform_buffers_memory.?[current_image], 0, @sizeOf(UniformBufferObject), .{});
        @memcpy(@as([*]u8, @ptrCast(data.?))[0..@sizeOf(UniformBufferObject)], std.mem.asBytes(&ubo));
        self.vkd.unmapMemory(self.device, self.uniform_buffers_memory.?[current_image]);
    }

    fn drawFrame(self: *Self) !void {
        _ = try self.vkd.waitForFences(self.device, (&self.in_flight_fences.?[self.current_frame])[0..1], .true, std.math.maxInt(u64));

        const result = self.vkd.acquireNextImageKHR(self.device, self.swap_chain, std.math.maxInt(u64), self.image_available_semaphores.?[self.current_frame], .null) catch |err| switch (err) {
            error.OutOfDateKHR => {
                try self.recreateSwapChain();
                return;
            },
            else => |e| return e,
        };

        if (result.result != .success and result.result != .suboptimal_khr) {
            return error.ImageAcquireFailed;
        }

        try self.updateUniformBuffer(self.current_frame);

        try self.vkd.resetFences(self.device, (&self.in_flight_fences.?[self.current_frame])[0..1]);

        try self.vkd.resetCommandBuffer(self.command_buffers.?[self.current_frame], .{});
        try self.recordCommandBuffer(self.command_buffers.?[self.current_frame], result.image_index);

        const wait_semaphores = [_]vk.Semaphore{self.image_available_semaphores.?[self.current_frame]};
        const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output = true }};
        const signal_semaphores = [_]vk.Semaphore{self.render_finished_semaphores.?[self.current_frame]};

        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = wait_semaphores.len,
            .p_wait_semaphores = &wait_semaphores,
            .p_wait_dst_stage_mask = &wait_stages,
            .command_buffer_count = 1,
            .p_command_buffers = @as([*]const vk.CommandBuffer, @ptrCast(&self.command_buffers.?[self.current_frame])),
            .signal_semaphore_count = signal_semaphores.len,
            .p_signal_semaphores = &signal_semaphores,
        };
        _ = try self.vkd.queueSubmit(self.graphics_queue, &[_]vk.SubmitInfo{submit_info}, self.in_flight_fences.?[self.current_frame]);

        const present_result = self.vkd.queuePresentKHR(self.present_queue, &.{
            .wait_semaphore_count = signal_semaphores.len,
            .p_wait_semaphores = @as([*]const vk.Semaphore, @ptrCast(&signal_semaphores)),
            .swapchain_count = 1,
            .p_swapchains = @as([*]const vk.SwapchainKHR, @ptrCast(&self.swap_chain)),
            .p_image_indices = @as([*]const u32, @ptrCast(&result.image_index)),
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
            .p_code = @as([*]const u32, @alignCast(@ptrCast(code))),
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
            const window_size = self.window.?.getFramebufferSize();

            return vk.Extent2D{
                .width = std.math.clamp(@as(u32, @intCast(window_size[0])), capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                .height = std.math.clamp(@as(u32, @intCast(window_size[1])), capabilities.min_image_extent.height, capabilities.max_image_extent.height),
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

        const supported_features = self.vki.getPhysicalDeviceFeatures(device);

        return indices.isComplete() and extensions_supported and swap_chain_adequate and supported_features.sampler_anisotropy == .true;
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

        for (queue_families, 0..) |queue_family, i| {
            if (indices.graphics_family == null and queue_family.queue_flags.graphics) {
                indices.graphics_family = @as(u32, @intCast(i));
            } else if (indices.present_family == null and (try self.vki.getPhysicalDeviceSurfaceSupportKHR(device, @as(u32, @intCast(i)), self.surface)) == .true) {
                indices.present_family = @as(u32, @intCast(i));
            }

            if (indices.isComplete()) {
                break;
            }
        }

        return indices;
    }

    fn getRequiredExtensions(allocator: Allocator) !std.array_list.Managed([*:0]const u8) {
        var extensions = std.array_list.Managed([*:0]const u8).init(allocator);
        try extensions.appendSlice(try glfw.getRequiredInstanceExtensions());

        if (enable_validation_layers) {
            try extensions.append(vk.instance_extensions.ext_debug_utils.name);
        }

        return extensions;
    }

    fn checkValidationLayerSupport(self: *Self) !bool {
        var layer_count: u32 = undefined;
        _ = try self.vkb.enumerateInstanceLayerProperties(&layer_count, null);

        const available_layers = try self.allocator.alloc(vk.LayerProperties, layer_count);
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

    fn debugCallback(_: vk.DebugUtilsMessageSeverityFlagsEXT, _: vk.DebugUtilsMessageTypeFlagsEXT, p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT, _: ?*anyopaque) callconv(vk.vulkan_call_conv) vk.Bool32 {
        if (p_callback_data != null) {
            std.log.debug("validation layer: {s}", .{p_callback_data.?.p_message orelse "unknown"});
        }

        return .false;
    }
};

pub fn main() void {
    var safe_allocator = std.heap.SafeAllocator.init(std.heap.page_allocator, .{});
    defer {
        const leaks = safe_allocator.deinit();
        if (leaks != 0) std.log.err("MemLeak: {d} allocations", .{leaks});
    }
    const allocator = safe_allocator.allocator();

    var app = HelloTriangleApplication.init(allocator) catch |err| {
        std.log.err("application failed to init with error: {any}", .{err});
        return;
    };
    defer app.deinit();
    app.run() catch |err| {
        std.log.err("application exited with error: {any}", .{err});
        return;
    };
}
