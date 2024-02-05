import "./style.css";
import stringTemplate from "./stringTemplate";
import {
  VariableDefinition,
  getSizeAndAlignmentOfUnsizedArrayElement,
  makeShaderDataDefinitions,
  makeStructuredView,
  makeTypedArrayViews,
  
} from "webgpu-utils";
import particleSimShader from "./shaders/particleSimShader";
import renderParticleShader from "./shaders/renderParticleShader";
import Shader from "./shaders/shader";
import { GUI } from "dat.gui";
import Stats from "stats.js";

declare var WebGPURecorder : any;

// new WebGPURecorder({
//   "frames": 100,
//   "export": "WebGPURecord",
//   "width": 800,
//   "height": 600,
//   "removeUnusedResources": false
// });



const PARTICLE_SIZE = 2;
const WORKGROUP_SIZE = 64;
const WORKGROUP_NUM = 16;
const PARTICLE_MAX_COUNT = WORKGROUP_SIZE * WORKGROUP_NUM;

const GRID_SIZE_IN_CELLS = [
  Math.sqrt(PARTICLE_MAX_COUNT),
  Math.sqrt(PARTICLE_MAX_COUNT),
];

const { device, canvasFormat, context, stats } = await setup();

const GRID_CELL_SIZE = context.canvas.width / GRID_SIZE_IN_CELLS[0];

const PARTICLE_SPEED = 25;

console.log(GRID_SIZE_IN_CELLS);
console.log(GRID_CELL_SIZE);

const particleComputeShader = new particleSimShader(
  WORKGROUP_SIZE,
  "Compute shader",
  device,
  GRID_SIZE_IN_CELLS
);
const particleRenderShader = new renderParticleShader(
  "Particle shader",
  device,
  GRID_SIZE_IN_CELLS
);

const particleStorageUnitSize = particleComputeShader.structs["Particle"].size;

console.log("ParticleStructSize : " + particleStorageUnitSize);

const particleStorageSizeInBytes = particleStorageUnitSize * PARTICLE_MAX_COUNT;

const gridStorageSizeInBytes = GRID_SIZE_IN_CELLS[0] * GRID_SIZE_IN_CELLS[1] * 4;

//#region Create Buffers

const particleStorageBuffer = device.createBuffer({
  label: "Particle storage buffer",
  size: particleStorageSizeInBytes,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const particleHeadsStorageBuffer = device.createBuffer({
  label: "Particle heads storage buffer",
  size: PARTICLE_MAX_COUNT * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const particleListsStorageBuffer = device.createBuffer({
  label: "Particle lists storage buffer",
  size: PARTICLE_MAX_COUNT * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

//#endregion


const particleView = makeStructuredView(
  particleComputeShader.storages["particles"],
  new ArrayBuffer(PARTICLE_MAX_COUNT * particleStorageUnitSize)
);

var seed = 1;
const random = () => {
    var x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

for (let i = 0; i < PARTICLE_MAX_COUNT; ++i) {
  const angle = rand() * 2 * Math.PI;
  
  particleView.views[i].oldPosition.set([
    context.canvas.width / 2,
    context.canvas.height / 2,
  ]);
  
  //particleView.views[i].oldPosition.set([rand(0, context.canvas.width), rand(0, context.canvas.height)]);
  const speed = PARTICLE_SPEED;
  particleView.views[i].velocity.set([
    Math.cos(angle) * speed,
    Math.sin(angle) * speed,
  ]);
}

device.queue.writeBuffer(particleStorageBuffer, 0, particleView.arrayBuffer);

const { view: simulationUniforms, buffer: simulationUniformsBuffer } =
  makeUniformViewAndBuffer(particleComputeShader, "SimulationUniforms");

const { view: globalUniforms, buffer: globalUniformsBuffer } =
  makeUniformViewAndBuffer(particleComputeShader, "GlobalUniforms");

globalUniforms.set({
  canvasSize: [context.canvas.width, context.canvas.height],
  particleSize: PARTICLE_SIZE,
  gridCellSizeInPixels : GRID_CELL_SIZE
});

device.queue.writeBuffer(globalUniformsBuffer, 0, globalUniforms.arrayBuffer);

const ctx = new OffscreenCanvas(32, 32).getContext("2d")!;

const grd = ctx.createRadialGradient(16, 16, 7, 16, 16, 16);
grd.addColorStop(0, "rgba(255,255,255,255)");
grd.addColorStop(1, "rgba(255,255,255,0)");

// Draw a filled Rectangle
ctx.fillStyle = grd;
ctx.fillRect(0, 0, 32, 32);

// const texture = device.createTexture({
//   size: [32, 32],
//   format: "rgba8unorm",
//   usage:
//     GPUTextureUsage.TEXTURE_BINDING |
//     GPUTextureUsage.COPY_DST |
//     GPUTextureUsage.RENDER_ATTACHMENT,
// });

// device.queue.copyExternalImageToTexture(
//   { source: ctx.canvas, flipY: true },
//   { texture, premultipliedAlpha: true },
//   [32, 32]
// );


const particleBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: "read-only-storage",
      },
    },
  ],
});

const commonBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
      buffer: {
        type: "uniform",
      },
    },
  ],
});

const computeBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage",
      },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage",
      },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage",
      },
    }
  ],
});

const simulationBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "uniform",
      },
    }
  ],
});

const resetPipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [
      commonBindGroupLayout,
      computeBindGroupLayout,
      simulationBindGroupLayout
    ],
  }),
  compute: {
    module: particleComputeShader.module,
    entryPoint: "reset",
  },
});

const gridComputePipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [
      commonBindGroupLayout,
      computeBindGroupLayout,
      simulationBindGroupLayout
    ],
  }),
  compute: {
    module: particleComputeShader.module,
    entryPoint: "c0",
  },
});

const computePipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [
      commonBindGroupLayout,
      computeBindGroupLayout,
      simulationBindGroupLayout
    ],
  }),
  compute: {
    module: particleComputeShader.module,
    entryPoint: "c1",
  },
});

const renderBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      sampler: {
        type: "filtering",
      },
    },
    // {
    //   binding: 1,
    //   visibility: GPUShaderStage.FRAGMENT,
    //   texture: {
    //     sampleType: "float",
    //   },
    // },

  ],
});

const renderPipeline = device.createRenderPipeline({
  label: "Particle render pipeline",
  layout: device.createPipelineLayout({
    bindGroupLayouts: [
      commonBindGroupLayout,
      particleBindGroupLayout,
      renderBindGroupLayout,
    ],
  }),
  vertex: {
    module: particleRenderShader.module,
    entryPoint: "vertexMain",
  },
  fragment: {
    module: particleRenderShader.module,
    entryPoint: "fragmentMain",
    targets: [
      {
        format: canvasFormat,
        blend: {
          color: {
            srcFactor: "one",
            dstFactor: "one-minus-src-alpha",
            operation: "add",
          },
          alpha: {
            srcFactor: "one",
            dstFactor: "one-minus-src-alpha",
            operation: "add",
          },
        },
      },
    ],
  },
});

const commonBindGroup = device.createBindGroup({
  label: "Common bind group",
  layout: commonBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: globalUniformsBuffer },
    },
  ],
});


const simulationBindGroup = device.createBindGroup({
  label: "simulationBindGroup",
  layout: simulationBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: simulationUniformsBuffer },
    }
  ],
});

const bindGroup0_c = device.createBindGroup({
  label: "bindGroup0_c",
  layout: computeBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: particleStorageBuffer },
    },
    {
      binding: 1,
      resource: { buffer: particleHeadsStorageBuffer },
    },
    {
      binding: 2,
      resource: { buffer: particleListsStorageBuffer },
    }
  ],
});

const bindGroup0 = device.createBindGroup({
  label: "bindGroup0",
  layout: particleBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: particleStorageBuffer },
    },
  ],
});

const sampler = device.createSampler({
  minFilter: "linear",
  magFilter: "linear",
});

const renderBindGroup = device.createBindGroup({
  layout: renderBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: sampler,
    },
    // {
    //   binding: 1,
    //   resource: texture.createView(),
    // },

  ],
});

const renderPassDescriptor = {
  label: "our basic canvas renderPass",
  colorAttachments: [
    {
      clearValue: [0.3, 0.3, 0.3, 1],
      loadOp: "clear",
      storeOp: "store",
    },
  ],
};

let step = 0;
let oldTime = 0;
function update(time: number) {
  stats.begin();

  step++;

  const dt = (time - oldTime) / 1000;

  oldTime = time;

  // Set some values via set
  simulationUniforms.set({
    deltaTime: dt,
  });

  // Upload the data to the GPU
  device.queue.writeBuffer(
    simulationUniformsBuffer,
    0,
    simulationUniforms.arrayBuffer
  );

  const encoder = device.createCommandEncoder();

  compute(encoder, step);
  render(encoder, step);

  device.queue.submit([encoder.finish()]);

  stats.end();
  requestAnimationFrame(update);
}

requestAnimationFrame(update);

function compute(encoder: GPUCommandEncoder, step: number) {

  {
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, commonBindGroup);
    pass.setBindGroup(1, bindGroup0_c);
    pass.setBindGroup(2, simulationBindGroup);
    pass.setPipeline(resetPipeline);
    pass.dispatchWorkgroups(WORKGROUP_NUM);
    pass.setPipeline(gridComputePipeline);
    pass.dispatchWorkgroups(WORKGROUP_NUM);
    pass.setPipeline(computePipeline);
    pass.dispatchWorkgroups(WORKGROUP_NUM);
    pass.end();
  }
  
}

function render(encoder: GPUCommandEncoder, step: number) {
  const canvasTexture = context.getCurrentTexture();
  (renderPassDescriptor.colorAttachments[0] as any).view =
    canvasTexture.createView();

  const pass = encoder.beginRenderPass(
    renderPassDescriptor as GPURenderPassDescriptor
  );
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, commonBindGroup);
  pass.setBindGroup(1, bindGroup0);
  pass.setBindGroup(2, renderBindGroup);
  pass.draw(6, PARTICLE_MAX_COUNT); // 6 vertices
  pass.end();
}

async function setup() {
  const appElement = document.querySelector<HTMLDivElement>("#app")!;
  appElement.innerHTML = `
  <canvas id="overlay"></canvas>
  <canvas id="render"></canvas>
  `;

  const canvas = document.querySelector("#render");
  const overlay = document.querySelector("#overlay");

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


  const size = Math.min(appElement.clientWidth, appElement.clientHeight);

  canvas.width = size;
  canvas.height = size;

  overlay.width = size;
  overlay.height = size;

 


function drawBoard(){

    // Box width
    var bw = size;
    // Box height
    var bh = size;
    // Padding
    var p = 0;

    var cs = size / GRID_SIZE_IN_CELLS[0];

    let context = overlay.getContext("2d");
    for (var x = 0; x <= bw; x += cs) {
        context.moveTo(0.5 + x + p, p);
        context.lineTo(0.5 + x + p, bh + p);
    }

    for (var x = 0; x <= bh; x += cs) {
        context.moveTo(p, 0.5 + x + p);
        context.lineTo(bw + p, 0.5 + x + p);
    }
    context.strokeStyle = "black";
    context.stroke();
}

drawBoard();

  if (!context) {
    throw new Error("Yo! No Canvas GPU context found.");
  }
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  const gui = new GUI();

  var stats = new Stats();
  stats.showPanel(1); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);

  return { device, canvasFormat, context, stats };
}

function rand(min = 0, max = 1) {
  return min + Math.random() * (max - min);
}

function makeUniformViewAndBuffer(shader: Shader, structName: string) {
  const view = makeStructuredView(shader.structs[structName]);
  return {
    view,
    buffer: device.createBuffer({
      size: view.arrayBuffer.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  };
}


