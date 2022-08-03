const std = @import("std");

const glfw = @import("glfw");

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const HelloTriangleApplication = struct {
    const Self = @This();

    window: ?glfw.Window = null,

    pub fn init() Self {
        return Self{};
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
            .resizable = false,
        });
    }

    fn initVulkan(_: *Self) !void {}

    fn mainLoop(self: *Self) !void {
        while (!self.window.?.shouldClose()) {
            try glfw.pollEvents();
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.window != null) self.window.?.destroy();

        glfw.terminate();
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
