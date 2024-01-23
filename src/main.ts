import './style.css';
import shader from './shaders/shader.wgsl?raw';
import particleSimShader from './shaders/particleSim.wgsl?raw';
import particle from './shaders/particle.wgsl?raw';
import stringTemplate from './stringTemplate';

const { device, canvasFormat, context } = await setup();

const WORKGROUP_SIZE = 64;

const WORKGROUP_NUM = 2;

const PARTICLE_MAX_COUNT = WORKGROUP_SIZE * WORKGROUP_NUM;

const particleShaderModule = device.createShaderModule({
  label: "Particle shader",
  code: stringTemplate(shader, {particleStruct: particle})
});

const particleComputeShaderModule = device.createShaderModule({
  label:"Compute shader", 
  code: stringTemplate(particleSimShader, {particleStruct: particle, workgroupSize: WORKGROUP_SIZE})
});



const particleStorageUnitSize = 
4 * 2 + // position is 2 32bit floats (4bytes each) 
4 * 2;  // velocity is 2 32bit floats (4bytes each) 
const particleStorageSizeInBytes = particleStorageUnitSize * PARTICLE_MAX_COUNT;

const particleStorageBuffer0 = device.createBuffer({
  label: "Particle storage buffer 0",
  size: particleStorageSizeInBytes,
  usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const particleStorageBuffer1 = device.createBuffer({
  label: "Particle storage buffer 1",
  size: particleStorageSizeInBytes,
  usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const particleStorageValues = new Float32Array(particleStorageSizeInBytes / 4);

for (let i = 0; i < PARTICLE_MAX_COUNT; ++i) {
  const offset = particleStorageUnitSize / 4 * i;
  particleStorageValues.set([rand(), rand()], offset);
}

device.queue.writeBuffer(particleStorageBuffer0, 0, particleStorageValues);

const ctx = new OffscreenCanvas(32,32).getContext('2d')!;

const grd = ctx.createRadialGradient(16, 16, 12, 16, 16, 16);
grd.addColorStop(0, "white");
grd.addColorStop(1, "rgba(255,255,255,0)");

// Draw a filled Rectangle
ctx.fillStyle = grd;
ctx.fillRect(0, 0, 32, 32);

const texture = device.createTexture({
  size: [32, 32],
  format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING |
         GPUTextureUsage.COPY_DST |
         GPUTextureUsage.RENDER_ATTACHMENT,
});

device.queue.copyExternalImageToTexture(
  { source: ctx.canvas, flipY: true },
  { texture, premultipliedAlpha: true },
  [32, 32],
);

const particleBindGroupLayout = device.createBindGroupLayout({
  entries:[
    {
      binding:0,
      visibility : GPUShaderStage.VERTEX ,
      buffer : {
        type : 'read-only-storage'
      }
    },  
    {
      binding:1,
      visibility : GPUShaderStage.COMPUTE,
      buffer : {
        type : 'storage'
      }
    }
  ]
});

const computePipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts:[
      particleBindGroupLayout
    ]
  }),
  compute: {
    module : particleComputeShaderModule,
    entryPoint : "updateParticle"
  }
});

const renderPipeline = device.createRenderPipeline({
  label: "Particle render pipeline",
  layout: device.createPipelineLayout({
    bindGroupLayouts:[
      particleBindGroupLayout,
      device.createBindGroupLayout({
        entries: [
          {
            binding:0,
            visibility: GPUShaderStage.FRAGMENT,
            sampler:{
              type:"filtering"
            }
          },
          {
            binding:1,
            visibility: GPUShaderStage.FRAGMENT,
            texture:{
              sampleType:"float"
            }
          }
        ]
      })
    ]
  }),
  vertex: {
    module: particleShaderModule,
    entryPoint: "vertexMain"
  },
  fragment: {
    module: particleShaderModule,
    entryPoint: "fragmentMain",
    targets: [
      {
       format: canvasFormat,
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      },
    ],
  }
});

const bindGroup0 = device.createBindGroup({
  layout: particleBindGroupLayout,
  entries:[
    {
      binding:0,
      resource: {buffer: particleStorageBuffer0}
    },
    {
      binding:1,
      resource: {buffer: particleStorageBuffer1}
    }
  ]
});

const bindGroup1 = device.createBindGroup({
  layout: particleBindGroupLayout,
  entries:[
    {
      binding:0,
      resource: {buffer: particleStorageBuffer1}
    },
    {
      binding:1,
      resource: {buffer: particleStorageBuffer0}
    }
  ]
});

const sampler = device.createSampler({
  minFilter: 'linear',
  magFilter: 'linear',
});

const renderBindGroup = device.createBindGroup({
  layout: renderPipeline.getBindGroupLayout(1),
  entries:[
    {
      binding:0,
      resource: sampler
    },
    {
      binding:1,
      resource: texture.createView()
    }
  ]
})

const renderPassDescriptor = {
  label: 'our basic canvas renderPass',
  colorAttachments: [
    {
      clearValue: [0.3, 0.3, 0.3, 1],
      loadOp: 'clear',
      storeOp: 'store',
    },
  ],
};

let step = 0;
async function update(time: number): Promise<void> {

  step ++;
  const encoder = device.createCommandEncoder();
  
  compute(encoder, step);
  render(encoder, step);
  
  device.queue.submit([encoder.finish()]);

  requestAnimationFrame(update)
}

requestAnimationFrame(update)

function compute(encoder : GPUCommandEncoder, step: number) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, step % 2 == 0 ? bindGroup0 : bindGroup1);
  pass.dispatchWorkgroups(WORKGROUP_NUM);
  pass.end();
}


function render(encoder : GPUCommandEncoder, step: number) {

  const canvasTexture = context.getCurrentTexture();
    ((renderPassDescriptor.colorAttachments[0]) as any).view =
        canvasTexture.createView();

  const pass = encoder.beginRenderPass(renderPassDescriptor as GPURenderPassDescriptor);
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, step % 2 == 0 ? bindGroup0 : bindGroup1);
  pass.setBindGroup(1, renderBindGroup);
  pass.draw(6, PARTICLE_MAX_COUNT); // 6 vertices
  pass.end();
}

async function setup() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <canvas width="512" height="512"></canvas>
  </div>
`;
  const canvas = document.querySelector("canvas");

  if (!canvas) {
    throw new Error("Yo! No canvas found.");
  }

  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("Yo! No Canvas GPU context found.");
  }
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });
  return { device, canvasFormat, context };
}

function rand (min = 0, max = 1) {
  return min + Math.random() * (max - min);
};
