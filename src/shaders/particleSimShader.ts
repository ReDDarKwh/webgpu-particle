import Shader from "./shader";
import common from "./common.wgsl?raw";

export default class particleSimShader extends Shader {
  /**
   *
   */
  constructor(
    workgroupSize: number,
    label: string,
    device: GPUDevice,
    gridSize: number[]
  ) {
    super(
      /* wgsl */ `
            
        ${common}

        const gridSize = vec2u(${gridSize[0]}, ${gridSize[1]});

        const gridSizeI = vec2i(${gridSize[0]}, ${gridSize[1]});

        struct SimulationUniforms{
            deltaTime: f32
        }

        @group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
        @group(1) @binding(1) var<storage, read_write> particleHeads: array<atomic<i32>>;
        @group(1) @binding(2) var<storage, read_write> particleLists: array<i32>;
        
        @group(2) @binding(0) var<uniform> simulation : SimulationUniforms;

        fn getIndexFromGridPos(gridPos : vec2u) -> u32{

            return gridPos.x + gridPos.y * gridSize.x;
        }

        fn applyCollision(
            global_invocation_index: u32
        ){
            var p = particles[global_invocation_index];

            if(p.collisionOtherIndex != -1){

                let o = particles[p.collisionOtherIndex];

                if(o.collisionOtherIndex == i32(global_invocation_index)){

                    let v = (p.mass - o.mass)/(p.mass + o.mass) * p.oldVelocity + 
                            2 * o.mass / (p.mass + o.mass) * o.oldVelocity;
                    
                    p.velocity = v;
                    p.temp = min(1, p.temp + 0.06);
                    particles[global_invocation_index] = p;
                }
            }
        }

        @compute @workgroup_size(${workgroupSize}) fn reset(
            @builtin(workgroup_id) workgroup_id : vec3<u32>,
            @builtin(local_invocation_index) local_invocation_index: u32,
            @builtin(num_workgroups) num_workgroups: vec3<u32>
        ){

            let workgroup_index =  
            workgroup_id.x +
            workgroup_id.y * num_workgroups.x +
            workgroup_id.z * num_workgroups.x * num_workgroups.y;
        
            let global_invocation_index =
                workgroup_index * ${workgroupSize} +
                local_invocation_index;

            atomicStore(&particleHeads[global_invocation_index], -1);

            var p = particles[global_invocation_index];
            p.temp = max(0.1, p.temp - 0.2 * simulation.deltaTime);
            particles[global_invocation_index] = p;
        }
        
        @compute @workgroup_size(${workgroupSize}) fn c0(
            @builtin(workgroup_id) workgroup_id : vec3<u32>,
            @builtin(local_invocation_index) local_invocation_index: u32,
            @builtin(num_workgroups) num_workgroups: vec3<u32>
        ){

            let workgroup_index =  
            workgroup_id.x +
            workgroup_id.y * num_workgroups.x +
            workgroup_id.z * num_workgroups.x * num_workgroups.y;
        
            let global_invocation_index =
                workgroup_index * ${workgroupSize} +
                local_invocation_index;

            applyCollision(global_invocation_index);

            var p = particles[global_invocation_index];

            var velocity = p.velocity;
            var nextPos = p.oldPosition + velocity * simulation.deltaTime;
            
            var screenBounceX : f32 = 1;
            var screenBounceY : f32 = 1;
            
            if(nextPos.x > globals.canvasSize.x){
                screenBounceX = -1;
            } else if (nextPos.x < 0) {
                screenBounceX = -1;
            }
            
            if(nextPos.y > globals.canvasSize.y){
                screenBounceY = -1;
            } else if (nextPos.y < 0) {
                screenBounceY = -1;
            }

            velocity *= vec2f(screenBounceX, screenBounceY);

            nextPos = p.oldPosition + velocity * simulation.deltaTime;

            p.velocity = velocity;

            p.position = nextPos;
            p.oldPosition = nextPos;
            
            let gridCoords = vec2u(nextPos / globals.gridCellSizeInPixels);
            let gridId = getIndexFromGridPos(gridCoords);
            
            particleLists[global_invocation_index] = atomicExchange(&particleHeads[gridId], i32(global_invocation_index));

            particles[global_invocation_index] = p;
        }

        @compute @workgroup_size(${workgroupSize}) fn c1(
            @builtin(workgroup_id) workgroup_id : vec3<u32>,
            @builtin(local_invocation_index) local_invocation_index: u32,
            @builtin(num_workgroups) num_workgroups: vec3<u32>
        ){

            let workgroup_index =  
                workgroup_id.x +
                workgroup_id.y * num_workgroups.x +
                workgroup_id.z * num_workgroups.x * num_workgroups.y;
            
            let global_invocation_index =
                workgroup_index * ${workgroupSize} +
                local_invocation_index;

            var p = particles[global_invocation_index];
            p.collisionOtherIndex = -1;
            p.oldVelocity = p.velocity;

            let gridCoords = vec2i(p.position / globals.gridCellSizeInPixels);

            for(var y = -1; y < 2; y++)
            {
                for(var x = -1; x < 2; x++)
                {
                    let gridCoordsWithOffset = gridCoords + vec2i(x,y); 

                    if( gridCoordsWithOffset.x < 0 || 
                        gridCoordsWithOffset.x > gridSizeI.x - 1 || 
                        gridCoordsWithOffset.y < 0 || 
                        gridCoordsWithOffset.y > gridSizeI.y - 1)
                    {
                        continue;
                    }

                    let gridCoordsIndex = getIndexFromGridPos(vec2u(gridCoordsWithOffset));

                    var particleIndex = atomicLoad(&particleHeads[gridCoordsIndex]);

                    while(particleIndex >= 0){
                        
                        if(particleIndex != i32(global_invocation_index)){

                            let o = particles[particleIndex];
                            let d = p.position - o.position;
                            let l = length(d);
                            let r2 = globals.particleSize;

                            if(l < r2){

                                let overlap = r2 - l;
                                p.oldPosition += d/l * overlap;

                                p.collisionOtherIndex = particleIndex;
                                break;
                            }
                        }

                        particleIndex = particleLists[particleIndex];
                    }
                    if(p.collisionOtherIndex != -1){
                        break;
                    }
                }
                if(p.collisionOtherIndex != -1){
                    break;
                }
            }

            particles[global_invocation_index] = p;
        } 

        `,
      label,
      device
    );
  }
}
