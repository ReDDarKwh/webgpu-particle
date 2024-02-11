
struct Particle {
  position : vec2f,
  oldPosition : vec2f, 

  velocity : vec2f,
  oldVelocity : vec2f,
  
  color : vec2f,
  mass : f32,
};

struct GlobalUniforms{
    canvasSize : vec2f,
    particleSize : f32,
    gridCellSizeInPixels : vec2f
}

@group(0) @binding(0) var<uniform> globals : GlobalUniforms;


fn random (st : vec2f) -> f32 {
                return fract(sin(dot(st.xy,
                                     vec2(12.9898,78.233)))*
                    43758.5453123);
            }