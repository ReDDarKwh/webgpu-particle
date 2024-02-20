
struct Particle {
  nextPos : vec2f,
  pos : vec2f,
  vel : vec2f,
  nextVel: vec2f,
  acc : vec2f,
  collisionOtherIndex : i32,
  mass : f32,
  temp : f32,
};

struct GlobalUniforms{
    canvasSize : vec2f,
    particleSize : f32,
    gridCellSizeInPixels : vec2f,
    gridSize : vec2u,
    color1 : vec3f,
    color2 : vec3f
}

@group(0) @binding(0) var<uniform> globals : GlobalUniforms;

fn random (x : f32) -> f32 {
                return fract(sin(x) * 43758.5453123);
            }

fn randomRangeInt(x: f32, max: f32) -> u32{

  return u32(round(random(x) * max));
}