
struct Particle {
  position : vec2f,
  oldPosition : vec2f, 

  velocity : vec2f,
  oldVelocity : vec2f,
  
  collisionOtherIndex : i32,
  mass : f32,
  temp : f32
};

struct GlobalUniforms{
    canvasSize : vec2f,
    particleSize : f32,
    gridCellSizeInPixels : vec2f
}

@group(0) @binding(0) var<uniform> globals : GlobalUniforms;


fn random (x : f32) -> f32 {
                return fract(sin(x) * 43758.5453123);
            }

fn randomRangeInt(x: f32, max: f32) -> u32{

  return u32(round(random(x) * max));
}