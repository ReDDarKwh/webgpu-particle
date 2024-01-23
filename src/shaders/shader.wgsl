${particleStruct}

const SIZE : f32 = 0.05;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex : u32,
  @builtin(instance_index) instance : u32
) -> VSOutput {

  let verts = array(
  vec2f(-1, 1),
  vec2f(-1, -1),
  vec2f(1, 1),
  vec2f(1, -1),
  vec2f(1, 1),
  vec2f(-1, -1)
  );

  var vertexPosition = verts[vertexIndex];
  let pos = vertexPosition * SIZE - 1 + particles[instance].position * 2;
  
  var vsOut: VSOutput;
  vsOut.position = vec4f(pos, 0.0, 1.0);
  vsOut.texcoord = vertexPosition * 0.5 + 0.5;

  return vsOut;
}

@group(1) @binding(0) var s: sampler;
@group(1) @binding(1) var t: texture_2d<f32>;

@fragment
fn fragmentMain(vsOut: VSOutput) -> @location(0) vec4f {

  return textureSample(t, s, vsOut.texcoord);
}
