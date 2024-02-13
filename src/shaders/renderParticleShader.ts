import Shader from "./shader";
import common from "./common.wgsl?raw"

export default class renderParticleShader extends Shader{
    /**
     *
     */
    constructor(label : string, device : GPUDevice, gridSize: number[]) {
        super(/* wgsl */ `
            ${common}

            const gridSize = vec2u(${gridSize[0], gridSize[1]});

            struct VSOutput {
            @builtin(position) position: vec4f,
            @location(0) texcoord: vec2f,
            @location(1) color: vec3f
            };

            @group(1) @binding(0) var<storage, read> particles: array<Particle>;

            @vertex
            fn vertexMain(
            @builtin(vertex_index) vertexIndex : u32,
    @builtin(instance_index) instance : u32
            ) -> VSOutput {

                let vertices = array(
                vec2f(-1, 1),
                vec2f(-1, -1),
                vec2f(1, 1),
                vec2f(1, -1),
                vec2f(1, 1),
                vec2f(-1, -1)
                );

                let vertexPosition = vertices[vertexIndex];
                let pos = (vertexPosition * globals.particleSize*3 + particles[instance].position) / globals.canvasSize * 2 - 1;

                var vsOut: VSOutput;
                vsOut.position = vec4f(pos, 0.0, 1.0);
                vsOut.texcoord = vertexPosition * 0.5 + 0.5;

                let color1 =  vec3f(1, 0, 1);
                let color2 = vec3f(0, 0, 0);

                vsOut.color = mix(color1, color2, 1 - particles[instance].temp);

                return vsOut;
            }
            
            @group(2) @binding(0) var s: sampler;
            @group(2) @binding(1) var t: texture_2d<f32>;

            @fragment
            fn fragmentMain(vsOut: VSOutput) -> @location(0) vec4f {

                let color = textureSample(t, s, vsOut.texcoord) * vec4f(vsOut.color,1);
                return color;
            }
        `, label, device);
    }
}