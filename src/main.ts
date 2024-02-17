import "./style.css";
import { makeStructuredView } from "webgpu-utils";
import particleSimShader from "./shaders/particleSimShader";
import renderParticleShader from "./shaders/renderParticleShader";
import Shader from "./shaders/shader";
import { GUI } from "dat.gui";
import Stats from "stats.js";


const { device, canvasFormat, context, stats, settings } =
  await setup();

const WORKGROUP_SIZE = 60;
let WORKGROUP_NUM: number;
let PARTICLE_MAX_COUNT: number;

const particleComputeShader = new particleSimShader(
  WORKGROUP_SIZE,
  "Compute shader",
  device
);
const particleRenderShader = new renderParticleShader(
  "Particle shader",
  device
);

//#region Create Buffers

const { view: simulationUniforms, buffer: simulationUniformsBuffer } =
  makeUniformViewAndBuffer(particleComputeShader, "SimulationUniforms");

const { view: staticSimulationUniforms, buffer: staticSimulationUniformsBuffer } =
makeUniformViewAndBuffer(particleComputeShader, "StaticSimulationUniforms");

const { view: globalUniforms, buffer: globalUniformsBuffer } =
  makeUniformViewAndBuffer(particleComputeShader, "GlobalUniforms");

const { view: renderUniforms, buffer: renderUniformsBuffer } =
  makeUniformViewAndBuffer(particleRenderShader, "RenderUniforms");

const texture = createParticleTexture();
const layouts = createLayouts();

const {
  commonBindGroupLayout,
  computeBindGroupLayout,
  simulationBindGroupLayout,
  particleBindGroupLayout,
  renderBindGroupLayout,
} = layouts;

const { resetPipeline, gridComputePipeline, computePipeline, renderPipeline } =
  createPipelines(layouts);

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
    },
    {
      binding: 1,
      resource: { buffer: staticSimulationUniformsBuffer },
    },
  ],
});

let bindGroups: {
  bindGroup0_c: GPUBindGroup | null;
  bindGroup0: GPUBindGroup | null;
} = {
  bindGroup0: null,
  bindGroup0_c: null,
};

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
    {
      binding: 1,
      resource: texture.createView(),
    },
    {
      binding: 2,
      resource: { buffer: renderUniformsBuffer },
    },
  ],
});

const renderPassDescriptor = {
  label: "our basic canvas renderPass",
  colorAttachments: [
    {
      clearValue: [...hexToRgb("#000000"), 1],
      loadOp: "clear",
      storeOp: "store",
    },
  ],
};

updateParticleCount();
updateRenderUniforms();
updateStaticSimulationUniforms();

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

  compute(encoder);
  render(encoder);

  device.queue.submit([encoder.finish()]);

  stats.end();
  requestAnimationFrame(update);
}

requestAnimationFrame(update);

function compute(encoder: GPUCommandEncoder) {
  {
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, commonBindGroup);
    pass.setBindGroup(1, bindGroups.bindGroup0_c);
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

function render(encoder: GPUCommandEncoder) {
  const canvasTexture = context.getCurrentTexture();
  (renderPassDescriptor.colorAttachments[0] as any).view =
    canvasTexture.createView();
  
  (renderPassDescriptor.colorAttachments[0] as any).clearValue = [...hexToRgb(settings.backgroundColor), 1];

  const pass = encoder.beginRenderPass(
    renderPassDescriptor as GPURenderPassDescriptor
  );
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, commonBindGroup);
  pass.setBindGroup(1, bindGroups.bindGroup0);
  pass.setBindGroup(2, renderBindGroup);
  pass.draw(6, PARTICLE_MAX_COUNT); // 6 vertices
  pass.end();
}

function createPipelines({
  commonBindGroupLayout,
  computeBindGroupLayout,
  simulationBindGroupLayout,
  particleBindGroupLayout,
  renderBindGroupLayout,
}: {
  commonBindGroupLayout: GPUBindGroupLayout;
  computeBindGroupLayout: GPUBindGroupLayout;
  simulationBindGroupLayout: GPUBindGroupLayout;
  particleBindGroupLayout: GPUBindGroupLayout;
  renderBindGroupLayout: GPUBindGroupLayout;
}) {
  const resetPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [
        commonBindGroupLayout,
        computeBindGroupLayout,
        simulationBindGroupLayout,
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
        simulationBindGroupLayout,
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
        simulationBindGroupLayout,
      ],
    }),
    compute: {
      module: particleComputeShader.module,
      entryPoint: "c1",
    },
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
  return {
    resetPipeline,
    gridComputePipeline,
    computePipeline,
    renderPipeline,
  };
}

function createLayouts() {
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
      },
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
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
    ],
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
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: "float",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  return {
    commonBindGroupLayout,
    computeBindGroupLayout,
    simulationBindGroupLayout,
    particleBindGroupLayout,
    renderBindGroupLayout,
  };
}

function createParticleTexture() {
  const ctx = new OffscreenCanvas(32, 32).getContext("2d")!;

  const grd = ctx.createRadialGradient(16, 16, 15, 16, 16, 16);
  grd.addColorStop(0, "rgba(255,255,255,255)");
  grd.addColorStop(1, "rgba(255,255,255,0)");

  // Draw a filled Rectangle
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 32, 32);

  const texture = device.createTexture({
    size: [32, 32],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: ctx.canvas, flipY: true },
    { texture, premultipliedAlpha: true },
    [32, 32]
  );
  return texture;
}

function updateParticleCount() {
  WORKGROUP_NUM = Math.ceil(settings.particleCount / WORKGROUP_SIZE);

  PARTICLE_MAX_COUNT = WORKGROUP_SIZE * WORKGROUP_NUM;

  const GRID_SIZE_IN_CELLS = GetGridSize({
    particleCount: PARTICLE_MAX_COUNT,
    canvasWidth: context.canvas.width,
    canvasHeight: context.canvas.height,
  });

  const GRID_CELL_SIZE_X = context.canvas.width / GRID_SIZE_IN_CELLS[0];
  const GRID_CELL_SIZE_Y = context.canvas.height / GRID_SIZE_IN_CELLS[1];

  console.log(GRID_CELL_SIZE_X, GRID_CELL_SIZE_Y);

  const particleStorageSizeInBytes =
    particleComputeShader.structs["Particle"].size * PARTICLE_MAX_COUNT;

  globalUniforms.set({
    canvasSize: [context.canvas.width, context.canvas.height],
    particleSize: settings.particleSize,
    gridCellSizeInPixels: [GRID_CELL_SIZE_X, GRID_CELL_SIZE_Y],
    gridSize: GRID_SIZE_IN_CELLS,
  });

  device.queue.writeBuffer(globalUniformsBuffer, 0, globalUniforms.arrayBuffer);

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

  const particleView = makeStructuredView(
    particleComputeShader.storages["particles"],
    new ArrayBuffer(particleStorageSizeInBytes)
  );

  for (let i = 0; i < PARTICLE_MAX_COUNT; ++i) {
    const angle = rand() * 2 * Math.PI;

    switch(settings.startingPosition){
      case "random" :
        particleView.views[i].oldPosition.set([
          rand(0, context.canvas.width),
          rand(0, context.canvas.height),
        ]);
      break;
      case "ring" :
        particleView.views[i].oldPosition.set([
          context.canvas.width / 2,
          context.canvas.height / 2,
        ]);
      break;
    }

    particleView.views[i].mass.set([rand(settings.minMass, settings.maxMass)]);
    particleView.views[i].collisionOtherIndex.set([-1]);

    const speed = settings.speed;
    particleView.views[i].velocity.set([
      Math.cos(angle) * speed,
      Math.sin(angle) * speed,
    ]);
  }

  device.queue.writeBuffer(particleStorageBuffer, 0, particleView.arrayBuffer);

  bindGroups.bindGroup0_c = device.createBindGroup({
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
      },
    ],
  });

  bindGroups.bindGroup0 = device.createBindGroup({
    label: "bindGroup0",
    layout: particleBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: particleStorageBuffer },
      },
    ],
  });
}

function hexToRgb(hex: string) {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16) / 255,
        parseInt(result[2], 16) / 255,
        parseInt(result[3], 16) / 255,
      ]
    : [0, 0, 0];
}

async function setup() {
  const appElement = document.querySelector<HTMLDivElement>("#app")!;
  appElement.innerHTML = `
  <canvas id="overlay"></canvas>
  <canvas id="render"></canvas>
  `;

  const canvas = document.querySelector("#render") as HTMLCanvasElement;
  const overlay = document.querySelector("#overlay") as HTMLCanvasElement;

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

  canvas.width = appElement.clientWidth;
  canvas.height = appElement.clientHeight;

  overlay.width = appElement.clientWidth;
  overlay.height = appElement.clientHeight;

  const overlayContext = overlay.getContext("2d")!;

  if (!context) {
    throw new Error("Yo! No Canvas GPU context found.");
  }
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  const gui = new GUI();

  const settings = {
    particleCount: 60 * 100,
    update: () => {},
    gridSizeX: 0,
    gridSizeY: 0,
    speed: 10,
    color1: "#FFFFFF",
    color2: "#000000",
    backgroundColor: "#000000",
    tempOnHit: 0.6,
    cooldownRate: 0.3,
    particleSize : 1,
    minMass: 1,
    maxMass: 10,
    startingPosition : "random",
    restart: updateParticleCount
  };
  
  gui.useLocalStorage = true;
  gui.remember(settings);
  gui.add(settings, "particleCount", 0, undefined, 60).onChange(updateParticleCount);
  gui.add(settings, "speed").onChange(updateParticleCount);
  gui.add(settings, "particleSize").onChange(updateParticleCount);
  gui.add(settings, "minMass").onChange(updateParticleCount);
  gui.add(settings, "maxMass").onChange(updateParticleCount);
  gui.addColor(settings, "color1").onChange(updateRenderUniforms);
  gui.addColor(settings, "color2").onChange(updateRenderUniforms);
  gui.addColor(settings, "backgroundColor");
  gui.add(settings,"tempOnHit").onChange(updateStaticSimulationUniforms);
  gui.add(settings,"cooldownRate").onChange(updateStaticSimulationUniforms);
  gui.add(settings, "startingPosition", ["random", "ring"]).onChange(updateParticleCount);
  gui.add(settings, "restart");

  var stats = new Stats();
  stats.showPanel(1); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);

  return { device, canvasFormat, context, overlayContext, stats, settings };
}

function updateRenderUniforms(){

  renderUniforms.set({
    color1: hexToRgb(settings.color1),
    color2: hexToRgb(settings.color2)
  });

  device.queue.writeBuffer(renderUniformsBuffer, 0, renderUniforms.arrayBuffer);
}

function updateStaticSimulationUniforms(){

  staticSimulationUniforms.set({
    tempOnHit: settings.tempOnHit,
    cooldownRate: settings.cooldownRate
  });

  device.queue.writeBuffer(staticSimulationUniformsBuffer, 0, staticSimulationUniforms.arrayBuffer);
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

function GetGridSize({
  particleCount,
  canvasWidth,
  canvasHeight,
}: {
  particleCount: number;
  canvasWidth: number;
  canvasHeight: number;
}): number[] {
  const factors = (number: number) =>
    [...Array(number + 1).keys()].filter((i) => number % i === 0);

  const ratio = (w: number, h: number) => Math.max(w, h) / Math.min(w, h);

  const cr = ratio(canvasWidth, canvasHeight);

  const f = factors(particleCount);
  const sf = f.slice(0, Math.ceil(f.length / 2));

  const c = sf
    .map((x, i) => [x, f[f.length - 1 - i]])
    .reduce((a, b) =>
      Math.abs(ratio(a[0], a[1]) - cr) < Math.abs(ratio(b[0], b[1]) - cr)
        ? a
        : b
    );

  return canvasWidth > canvasHeight
    ? [Math.max(c[0], c[1]), Math.min(c[0], c[1])]
    : [Math.min(c[0], c[1]), Math.max(c[0], c[1])];
}


