const std = @import("std");

const deps = @import("deps.zig");
const glfw = deps.imports.build_glfw;
const vkgen = deps.imports.vk_gen;

pub fn build(b: *std.build.Builder) !void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    // Find all zig files in src and create run steps for each of the tutorials.
    {
        const src_dir = try std.fs.path.join(b.allocator, &.{ b.build_root, "src" }); //try std.fmt.allocPrint(b.allocator, "src/{s}", .{entry.name});
        defer b.allocator.free(src_dir);

        var dir = try std.fs.cwd().openDir(src_dir, .{ .iterate = true });
        defer dir.close();

        var itr = dir.iterate();
        while (try itr.next()) |entry| {
            switch (entry.kind) {
                .File => {
                    if (entry.name.len < 5 or !std.mem.endsWith(u8, entry.name, ".zig")) {
                        continue;
                    }
                    const file_location = try std.fs.path.join(b.allocator, &.{ src_dir, entry.name }); //try std.fmt.allocPrint(b.allocator, "src/{s}", .{entry.name});
                    defer b.allocator.free(file_location);
                    const run_txt = try std.fmt.allocPrint(b.allocator, "run-{s}", .{entry.name[0..2]});
                    defer b.allocator.free(run_txt);

                    const exe = b.addExecutable(entry.name[0 .. entry.name.len - 4], file_location);
                    exe.setTarget(target);
                    exe.setBuildMode(mode);

                    // mach-glfw
                    exe.addPackage(glfw.pkg);
                    glfw.link(b, exe, .{});

                    // vulkan-zig: Create a step that generates vk.zig (stored in zig-cache) from the provided vulkan registry.
                    const gen = vkgen.VkGenerateStep.init(b, deps.cache ++ "/git/github.com/Snektron/vulkan-zig/examples/vk.xml", "vk.zig");
                    exe.addPackage(gen.package);

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
