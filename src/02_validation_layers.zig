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

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};

const enable_validation_layers: bool = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    else => false,
};

const BaseDispatch = vk.BaseWrapper;

const InstanceDispatch = vk.InstanceWrapper;

const HelloTriangleApplication = struct {
    const Self = @This();
    allocator: Allocator,

    window: ?*glfw.Window = null,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,

    instance: vk.Instance = null_instance,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null,

    pub fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn run(self: *Self) !void {
        try self.initWindow();
        try self.initVulkan();
        try self.mainLoop();
    }

    fn initWindow(self: *Self) !void {
        try glfw.init();
        glfw.windowHint(.client_api, .no_api);
        glfw.windowHint(.resizable, false);
        self.window = try glfw.Window.create(WIDTH, HEIGHT, "Vulkan", null, null);
    }

    fn initVulkan(self: *Self) !void {
        try self.createInstance();
        try self.setupDebugMessenger();
    }

    fn mainLoop(self: *Self) !void {
        while (!self.window.?.shouldClose()) {
            glfw.pollEvents();
        }
    }

    pub fn deinit(self: *Self) void {
        if (enable_validation_layers and self.debug_messenger != .null) self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.debug_messenger, null);

        if (self.instance != null_instance) self.vki.destroyInstance(self.instance, null);

        if (self.window != null) self.window.?.destroy();

        glfw.terminate();
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

    var app = HelloTriangleApplication.init(allocator);
    defer app.deinit();
    app.run() catch |err| {
        std.log.err("application exited with error: {any}", .{err});
        return;
    };
}
