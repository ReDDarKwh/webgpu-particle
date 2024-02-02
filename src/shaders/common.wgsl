
struct Particle {
  position : vec2f,
  velocity : vec2f,
  gridCoords : vec2i,
  color : vec3f
};

struct GlobalUniforms{
    canvasSize : vec2f,
    particleSize : f32,
    gridCellSizeInPixels : u32
}

@group(0) @binding(0) var<uniform> globals : GlobalUniforms;