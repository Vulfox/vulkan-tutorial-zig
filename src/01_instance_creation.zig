const std = @import("std");

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

const BaseDispatch = vk.BaseWrapper;

const InstanceDispatch = vk.InstanceWrapper;

const HelloTriangleApplication = struct {
    const Self = @This();

    window: ?*glfw.Window = null,

    vkb: BaseDispatch = undefined,
    vki: InstanceDispatch = undefined,

    instance: vk.Instance = null_instance,

    pub fn init() Self {
        return Self{};
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
    }

    fn mainLoop(self: *Self) !void {
        while (!self.window.?.shouldClose()) {
            glfw.pollEvents();
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.instance != null_instance) self.vki.destroyInstance(self.instance, null);

        if (self.window != null) self.window.?.destroy();

        glfw.terminate();
    }

    fn createInstance(self: *Self) !void {
        const vk_proc = @as(*const fn (instance: ?vk.Instance, procname: [*:0]const u8) callconv(.c) vk.PfnVoidFunction, @ptrCast(&glfw.getInstanceProcAddress));
        self.vkb = BaseDispatch.load(vk_proc);

        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(1, 0, 0, 0).toU32(),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0, 0).toU32(),
            .api_version = vk.API_VERSION_1_2.toU32(),
        };

        const glfw_extensions = try glfw.getRequiredInstanceExtensions();

        const create_info = vk.InstanceCreateInfo{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = null,
            .enabled_extension_count = @as(u32, @intCast(glfw_extensions.len)),
            .pp_enabled_extension_names = glfw_extensions.ptr,
        };

        self.instance = try self.vkb.createInstance(&create_info, null);

        self.vki = InstanceDispatch.load(self.instance, vk_proc);
    }
};

pub fn main() void {
    var app = HelloTriangleApplication.init();
    defer app.deinit();
    app.run() catch |err| {
        std.log.err("application exited with error: {any}", .{err});
        return;
    };
}
