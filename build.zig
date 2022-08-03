const std = @import("std");

const deps = @import("deps.zig");
const glfw = deps.imports.build_glfw;
const vkgen = deps.imports.vk_gen;
const vkbuild = deps.imports.vk_build;

pub fn build(b: *std.build.Builder) !void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    // Find all zig files in src and create run steps for each of the tutorials.
    {
        const src_dir = try std.fs.path.join(b.allocator, &.{ b.build_root, "src" });

        var dir = try std.fs.cwd().openDir(src_dir, .{ .iterate = true });
        defer dir.close();

        // generate vulkan package
        const gen = vkgen.VkGenerateStep.init(b, deps.cache ++ "/git/github.com/Snektron/vulkan-zig/examples/vk.xml", "vk.zig");

        // shader resources, to be compiled using glslc
        const shaders = vkbuild.ResourceGenStep.init(b, "resources.zig");
        shaders.addShader("vert_09", "src/09_shader_base.vert");
        shaders.addShader("frag_09", "src/09_shader_base.frag");
        shaders.addShader("vert_18", "src/18_shader_vertexbuffer.vert");
        shaders.addShader("frag_18", "src/18_shader_vertexbuffer.frag");

        var itr = dir.iterate();
        while (try itr.next()) |entry| {
            switch (entry.kind) {
                .File => {
                    if (entry.name.len < 5 or !std.mem.endsWith(u8, entry.name, ".zig")) {
                        continue;
                    }
                    const file_location = try std.fs.path.join(b.allocator, &.{ src_dir, entry.name });
                    defer b.allocator.free(file_location);
                    const run_txt = try std.fmt.allocPrint(b.allocator, "run-{s}", .{entry.name[0..2]});
                    defer b.allocator.free(run_txt);

                    const exe = b.addExecutable(entry.name[0 .. entry.name.len - 4], file_location);
                    exe.setTarget(target);
                    exe.setBuildMode(mode);

                    // mach-glfw
                    exe.addPackage(glfw.pkg);
                    glfw.link(b, exe, .{});

                    // vulkan-zig
                    exe.addPackage(gen.package);
                    exe.addPackage(shaders.package);

                    const run_cmd = exe.run();
                    run_cmd.step.dependOn(b.getInstallStep());
                    if (b.args) |args| {
                        run_cmd.addArgs(args);
                    }

                    const run_step = b.step(run_txt, "Run the numbered tutorial");

                    run_step.dependOn(&run_cmd.step);
                },
                else => continue,
            }
        }
    }
}
