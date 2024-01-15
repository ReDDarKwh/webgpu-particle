import './style.css';
import shader from './shaders/shader.wgsl?raw';
import particleSimShader from './shaders/particleSim.wgsl?raw';
import stringTemplate from './stringTemplate';

const { device, canvasFormat, context } = await Setup();

const particleShaderModule = device.createShaderModule({
  label: "Particle shader",
  code: stringTemplate(shader, {})
});

const particleComputeShaderModule = device.createShaderModule({
  label:"Compute shader", 
  code: stringTemplate(particleSimShader, {})
});

const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
      buffer: {
        type: "storage"
      }
    },
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    }
  ]
});

const pipelineLayout = device.createPipelineLayout({
  bindGroupLayouts:[
    bindGroupLayout
  ]
});

const computePipeline = device.createComputePipeline({
  layout: pipelineLayout,
  compute: {
    module : particleComputeShaderModule,
    entryPoint : "updateParticle"
  }
})

const renderPipeline = device.createRenderPipeline({
  label: "Cell pipeline",
  layout: pipelineLayout,
  vertex: {
    module: particleShaderModule,
    entryPoint: "vertexMain"
  },
  fragment: {
    module: particleShaderModule,
    entryPoint: "fragmentMain",
    targets: [{
      format: canvasFormat
    }]
  }
});

const bindGroup = device.createBindGroup({
  label: "Cell renderer bind group",
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [],
});

async function update(time: number): Promise<void> {
  
  const encoder = device.createCommandEncoder();
  
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context!.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: [0.15, 0.1, 0.2, 1],
      storeOp: "store",
    }]
  });
  
  pass.setPipeline(renderPipeline);
  pass.draw(6); // 6 vertices
  pass.end();
  
  device.queue.submit([encoder.finish()]);
  
  requestAnimationFrame(update)
}
requestAnimationFrame(update)

async function Setup() {
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
