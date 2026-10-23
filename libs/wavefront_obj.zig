const std = @import("std");
const Io = std.Io;

pub const Vertex = struct {
    position: usize,
    textureCoordinate: ?usize = null,
};

pub const Face = struct {
    vertices: []Vertex,
};

pub const Vec3 = struct { x: f32, y: f32, z: f32 };
pub const Vec2 = struct { x: f32, y: f32 };

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    positions: []Vec3,
    textureCoordinates: []Vec2,
    faces: []Face,

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Minimal Wavefront OBJ loader: reads positions, texture coordinates and
/// triangulates faces. Normals/materials are intentionally ignored because the
/// tutorials only need positions and UVs.
pub fn loadFile(allocator: std.mem.Allocator, path: []const u8) !Model {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();

    const data = try Io.Dir.cwd().readFileAlloc(
        Io.Threaded.global_single_threaded.io(),
        path,
        allocator,
        .limited(256 * 1024 * 1024),
    );
    defer allocator.free(data);

    var positions: std.ArrayList(Vec3) = .empty;
    var tex_coords: std.ArrayList(Vec2) = .empty;
    var faces: std.ArrayList(Face) = .empty;

    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        var tokens = std.mem.tokenizeScalar(u8, line, ' ');
        const tag = tokens.next() orelse continue;

        if (std.mem.eql(u8, tag, "v")) {
            const x = try std.fmt.parseFloat(f32, tokens.next() orelse return error.InvalidFormat);
            const y = try std.fmt.parseFloat(f32, tokens.next() orelse return error.InvalidFormat);
            const z = try std.fmt.parseFloat(f32, tokens.next() orelse return error.InvalidFormat);
            try positions.append(a, .{ .x = x, .y = y, .z = z });
        } else if (std.mem.eql(u8, tag, "vt")) {
            const x = try std.fmt.parseFloat(f32, tokens.next() orelse return error.InvalidFormat);
            const y = try std.fmt.parseFloat(f32, tokens.next() orelse return error.InvalidFormat);
            try tex_coords.append(a, .{ .x = x, .y = y });
        } else if (std.mem.eql(u8, tag, "f")) {
            var vertices: std.ArrayList(Vertex) = .empty;
            defer vertices.deinit(a);

            while (tokens.next()) |spec| {
                var parts = std.mem.splitScalar(u8, spec, '/');
                const position_text = parts.next() orelse return error.InvalidFormat;
                const position = try std.fmt.parseInt(usize, position_text, 10);

                var tex_coord: ?usize = null;
                if (parts.next()) |text| {
                    if (text.len != 0) tex_coord = try std.fmt.parseInt(usize, text, 10);
                }

                try vertices.append(a, .{
                    .position = position - 1,
                    .textureCoordinate = if (tex_coord) |t| t - 1 else null,
                });
            }

            if (vertices.items.len < 3) return error.InvalidFormat;

            // Triangulate polygons as a fan from the first vertex.
            var i: usize = 1;
            while (i + 1 < vertices.items.len) : (i += 1) {
                const triangle = try a.alloc(Vertex, 3);
                triangle[0] = vertices.items[0];
                triangle[1] = vertices.items[i];
                triangle[2] = vertices.items[i + 1];
                try faces.append(a, .{ .vertices = triangle });
            }
        }
    }

    if (positions.items.len == 0 or tex_coords.items.len == 0 or faces.items.len == 0) {
        return error.InvalidFormat;
    }

    return .{
        .arena = arena,
        .positions = try positions.toOwnedSlice(a),
        .textureCoordinates = try tex_coords.toOwnedSlice(a),
        .faces = try faces.toOwnedSlice(a),
    };
}
