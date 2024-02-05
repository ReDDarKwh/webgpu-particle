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

        const gridSize = vec2u(${(gridSize[0], gridSize[1])});

        struct SimulationUniforms{
            deltaTime: f32
        }

        @group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
        @group(1) @binding(1) var<storage, read_write> particleHeads: array<atomic<i32>>;
        @group(1) @binding(2) var<storage, read_write> particleLists: array<i32>;
        
        @group(2) @binding(0) var<uniform> simulation : SimulationUniforms;

        fn wrap(vp : f32, mp : f32) -> f32 { 
            
            var v = vp;
            var m = mp;

            if (m > 0) {
              if (v < 0) {v = m-((-v) % m);}          // get negative v into region [0, m)
            } else {
              m = -m;                               // the positive value is easier to work with
              if (v < 0) {v += m * f32(1 + i32(-v/m));}  // add m enough times so v > 0
              v =  m - (v % m);                     // get v % m, then flip 
            }
            return v % m;                           // return v % m, now that both are positive
        }

        fn getIndexFromGridPos(gridPos : vec2f) -> i32{

            let floatGridSize = vec2f(gridSize);
            let wrapped = vec2f(gridPos.x, gridPos.y);

            if(gridPos.x < 0 || gridPos.x >= floatGridSize.x || gridPos.y < 0 || gridPos.y >= floatGridSize.y){
                return -1;
            }

            return i32(wrapped.x + wrapped.y * floatGridSize.x);
        }

        fn collideParticles(particle: ptr<function, Particle>, otherParticle: Particle){

            let v = (*particle).position - otherParticle.position;
            let d = length(v);

            if(d < globals.particleSize * 2){
                (*particle).color = vec2f(1,0);
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

            particles[global_invocation_index].color = vec2f(0,1);
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

            var p = particles[global_invocation_index];

            var nextPos = p.oldPosition + p.velocity * simulation.deltaTime;
            
            var screenWarpVec = vec2f(0,0);
            
            let particleSizeTimes2 = globals.particleSize * 4;

            if(nextPos.x > globals.canvasSize.x - particleSizeTimes2){
                particles[global_invocation_index].velocity *= vec2f(-1,0);
            } else if (nextPos.x < particleSizeTimes2) {
                particles[global_invocation_index].velocity *= vec2f(-1,0);
            }
            
            if(nextPos.y > globals.canvasSize.y - particleSizeTimes2){
                particles[global_invocation_index].velocity *= vec2f(0,-1);
            } else if (nextPos.y < particleSizeTimes2) {
                particles[global_invocation_index].velocity *= vec2f(0,-1);
            }

            nextPos += screenWarpVec;

            particles[global_invocation_index].position = nextPos;
            particles[global_invocation_index].oldPosition = nextPos;
            
            let gridCoords = nextPos / f32(globals.gridCellSizeInPixels);
            let gridId = getIndexFromGridPos(gridCoords);
            if(gridId != -1){
                particleLists[global_invocation_index] = atomicExchange(&particleHeads[gridId], i32(global_invocation_index));
            }
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


            let gridCoords = particles[global_invocation_index].oldPosition / f32(globals.gridCellSizeInPixels);

            var collide = 0;

            for(var y = -1; y < 3; y++)
            {
                for(var x = -1; x < 3; x++)
                {
                    let gridCoordsWithOffset = gridCoords + vec2f(f32(x),f32(y)); 
                    let gridCoordsIndex = getIndexFromGridPos(gridCoordsWithOffset);

                    if(gridCoordsIndex == -1){
                        continue;
                    }

                    var particleIndex = atomicLoad(&particleHeads[gridCoordsIndex]);

                    while(particleIndex >= 0){
                        
                        if(particleIndex != i32(global_invocation_index)){

                            let otherParticle = particles[particleIndex];

                            let v = particles[global_invocation_index].position - otherParticle.position;
                            let d = length(v);

                            if(d < globals.particleSize * 2){
                                collide = 1;
                                break;
                            }
                        }

                        particleIndex = particleLists[particleIndex];
                    }
                }
            }

            if(collide == 1){
                particles[global_invocation_index].color = vec2f(1,0);
            }

            //storageBarrier();
            
        } 
        `,
      label,
      device
    );
  }
}
