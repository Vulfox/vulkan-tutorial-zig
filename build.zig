const std = @import("std");
const Io = std.Io;

const deps = @import("deps.zig");

// Source-only dependencies live in the `zigmod fetch` cache. Populate it with
// `$ zigmod fetch` before running `zig build`.
const dep = struct {
    const zglfw = deps.cache ++ "/git/github.com/zig-gamedev/zglfw";
    const system_sdk = deps.cache ++ "/git/github.com/zig-gamedev/system_sdk";
    const vulkan_zig = deps.cache ++ "/git/github.com/Snektron/vulkan-zig";
    const vulkan_headers = deps.cache ++ "/git/github.com/KhronosGroup/Vulkan-Headers";
    const stb = deps.cache ++ "/git/github.com/nothings/stb";
    const zalgebra = deps.cache ++ "/git/github.com/kooparse/zalgebra";
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // A path pointing outside of the build root; used for zigmod-owned sources.
    const ext = struct {
        fn path(sub: []const u8) std.Build.LazyPath {
            return .{ .cwd_relative = sub };
        }
    }.path;

    // mach-glfw went away with the maintenance wind-down of the Hexops project;
    // its zglfw successor (zig-gamedev/zglfw) acts as the drop-in replacement.
    // zigmod only wraps plain zig sources, so GLFW's C compilation and the
    // `src/zglfw.zig` module assembly happen here, mirroring zglfw's build.zig.
    const zglfw_options_mod = blk: {
        const options = b.addOptions();
        options.addOption(bool, "shared", false);
        options.addOption(bool, "enable_x11", true);
        options.addOption(bool, "enable_wayland", true);
        options.addOption(bool, "enable_vulkan_import", false);
        break :blk options.createModule();
    };

    const glfw_c_module = blk: {
        const translate_c = b.addTranslateC(.{
            .root_source_file = ext(dep.zglfw ++ "/src/c.h"),
            .target = target,
            .optimize = optimize,
        });
        translate_c.addIncludePath(ext(dep.zglfw ++ "/libs/glfw/include"));
        translate_c.addSystemIncludePath(ext(dep.system_sdk ++ "/linux/include"));
        translate_c.addSystemIncludePath(ext(dep.system_sdk ++ "/linux/include/wayland"));
        break :blk translate_c.createModule();
    };

    const zalgebra_module = b.createModule(.{
        .root_source_file = ext(dep.zalgebra ++ "/src/main.zig"),
    });

    // stb_image is a single-header C library; translate the header so Zig code
    // can call stbi_load/stbi_image_free.
    const stb_module = blk: {
        const translate_c = b.addTranslateC(.{
            .root_source_file = ext(dep.stb ++ "/stb_image.h"),
            .target = target,
            .optimize = optimize,
        });
        translate_c.addIncludePath(ext(dep.stb));
        break :blk translate_c.createModule();
    };

    // Small built-in OBJ loader (the original wavefront-obj dependency is no
    // longer available).
    const wavefront_obj_module = b.createModule(.{
        .root_source_file = b.path("libs/wavefront_obj.zig"),
    });

    const zglfw_module = b.createModule(.{
        .root_source_file = ext(dep.zglfw ++ "/src/zglfw.zig"),
        .imports = &.{
            .{ .name = "zglfw_options", .module = zglfw_options_mod },
            .{ .name = "c", .module = glfw_c_module },
        },
    });

    const glfw_lib = b.addLibrary(.{
        .name = "glfw",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    {
        const incl = &.{
            "/libs/glfw/include",
            "/libs/glfw/src/wayland",
        };
        inline for (incl) |item| {
            glfw_lib.root_module.addIncludePath(ext(dep.zglfw ++ item));
        }
        glfw_lib.root_module.addSystemIncludePath(ext(dep.system_sdk ++ "/linux/include"));
        glfw_lib.root_module.addSystemIncludePath(ext(dep.system_sdk ++ "/linux/include/wayland"));

        const src_dir = dep.zglfw ++ "/libs/glfw/src";
        const src_dir_path = ext(src_dir);
        glfw_lib.root_module.addCSourceFiles(.{
            .root = src_dir_path,
            .files = &.{
                "platform.c",
                "monitor.c",
                "init.c",
                "vulkan.c",
                "input.c",
                "context.c",
                "window.c",
                "osmesa_context.c",
                "egl_context.c",
                "null_init.c",
                "null_monitor.c",
                "null_window.c",
                "null_joystick.c",
                "posix_time.c",
                "posix_thread.c",
                "posix_module.c",
                "xkb_unicode.c",
                "linux_joystick.c",
                "posix_poll.c",
                "x11_init.c",
                "x11_monitor.c",
                "x11_window.c",
                "glx_context.c",
                "wl_init.c",
                "wl_monitor.c",
                "wl_window.c",
            },
            .flags = &.{ "-D_GLFW_X11=1", "-D_GLFW_WAYLAND=1" },
        });
        glfw_lib.root_module.linkSystemLibrary("X11", .{});
    }

    // Build vulkan-zig's generator and run it against the Vulkan XML registry
    // to produce the "vulkan" module.
    const vk_generator_exe = b.addExecutable(.{
        .name = "vulkan-zig-generator",
        .root_module = b.createModule(.{
            .root_source_file = ext(dep.vulkan_zig ++ "/src/main.zig"),
            .target = b.graph.host,
        }),
    });
    const vk_generate_cmd = b.addRunArtifact(vk_generator_exe);
    vk_generate_cmd.addFileArg(ext(dep.vulkan_headers ++ "/registry/vk.xml"));
    const vk_zig = vk_generate_cmd.addOutputFileArg("vk.zig");
    const vulkan_module = b.createModule(.{
        .root_source_file = vk_zig,
    });

    // Compile the tutorial shaders with glslc and wrap the resulting SPIR-V
    // bytecode in a "resources" module, à la vulkan-zig's old ResourceGenStep.
    const resources_module = blk: {
        const shaders = [_]struct { name: []const u8, source: []const u8 }{
            .{ .name = "vert_09", .source = "src/09_shader_base.vert" },
            .{ .name = "frag_09", .source = "src/09_shader_base.frag" },
            .{ .name = "vert_18", .source = "src/18_shader_vertexbuffer.vert" },
            .{ .name = "frag_18", .source = "src/18_shader_vertexbuffer.frag" },
            .{ .name = "vert_22", .source = "src/22_shader_ubo.vert" },
            .{ .name = "frag_22", .source = "src/22_shader_ubo.frag" },
            .{ .name = "vert_26", .source = "src/26_shader_textures.vert" },
            .{ .name = "frag_26", .source = "src/26_shader_textures.frag" },
            .{ .name = "vert_27", .source = "src/27_shader_depth.vert" },
            .{ .name = "frag_27", .source = "src/27_shader_depth.frag" },
        };

        const generated = b.addWriteFiles();
        var resources: std.ArrayList(u8) = .empty;
        for (shaders) |shader| {
            const cmd = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.2", "-o" });
            const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{shader.name}));
            cmd.addFileArg(b.path(shader.source));

            _ = generated.addCopyFile(
                spv,
                b.fmt("{s}.spv", .{shader.name}),
            );
            resources.appendSlice(
                b.allocator,
                b.fmt("const {s}_data align(4) = @embedFile(\"{s}.spv\").*;\npub const {s}: []const u8 = &{s}_data;\n", .{ shader.name, shader.name, shader.name, shader.name }),
            ) catch @panic("OOM");
        }

        const resources_zig = generated.add("resources.zig", resources.items);
        break :blk b.createModule(.{
            .root_source_file = resources_zig,
        });
    };

    // Find all zig files in src and create run steps for each of the tutorials.
    // Build logic reads the directory, so let `zig build` know to re-run the
    // configure phase whenever a file appears or disappears in `src`.
    b.dependOnDirectoryContents(b.path("src"));
    b.dependOnDirectoryContents(b.path("resources"));
    {
        const root_path = b.root.root_dir.path orelse "";
        const src_sub_path = if (root_path.len == 0)
            "src"
        else
            b.fmt("{s}/src", .{root_path});

        const src_dir = Io.Dir.cwd().openDir(b.graph.io, src_sub_path, .{ .iterate = true }) catch |err|
            @panic(b.fmt("failed to open the src directory: {s}", .{@errorName(err)}));
        defer src_dir.close(b.graph.io);

        var itr = src_dir.iterate();
        while (itr.next(b.graph.io) catch |err|
            @panic(b.fmt("failed to enumerate the src directory: {s}", .{@errorName(err)}))) |entry|
        {
            switch (entry.kind) {
                .file => {},
                else => continue,
            }
            if (entry.name.len < 5 or !std.mem.endsWith(u8, entry.name, ".zig")) {
                continue;
            }

            const exe = b.addExecutable(.{
                .name = b.allocator.dupe(u8, entry.name[0 .. entry.name.len - 4]) catch @panic("OOM"),
                .root_module = b.createModule(.{
                    .root_source_file = b.path(b.fmt("src/{s}", .{entry.name})),
                    .target = target,
                    .optimize = optimize,
                    .link_libc = true,
                    .imports = &.{
                        .{ .name = "glfw", .module = zglfw_module },
                        .{ .name = "vulkan", .module = vulkan_module },
                        .{ .name = "resources", .module = resources_module },
                        .{ .name = "zalgebra", .module = zalgebra_module },
                        .{ .name = "stb", .module = stb_module },
                        .{ .name = "wavefront-obj", .module = wavefront_obj_module },
                    },
                }),
            });

            // stb image loading
            exe.root_module.addIncludePath(ext(dep.stb));
            exe.root_module.addCSourceFile(.{
                .file = b.path("libs/stb/stb_impl.c"),
                .flags = &.{"-std=c99"},
            });
            exe.root_module.linkLibrary(glfw_lib);

            const run_cmd = b.addRunArtifact(exe);
            run_cmd.step.dependOn(b.getInstallStep());
            // Tutorials open their artwork through cwd-relative paths.
            run_cmd.setCwd(.{ .cwd_relative = if (root_path.len == 0)
                "."
            else
                root_path });
            const run_step = b.step(b.fmt("run-{s}", .{entry.name[0..2]}), "Run the numbered tutorial");
            run_step.dependOn(&run_cmd.step);
        }
    }
}
