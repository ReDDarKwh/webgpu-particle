import Shader from "./shader";
import common from "./common.wgsl?raw";

export default class particleSimShader extends Shader {
  /**
   *
   */
  constructor(workgroupSize: number, label: string, device: GPUDevice, gridSize: number[]) {
    super(
      /* wgsl */ `
            
        ${common}

        const gridSize = vec2u(${gridSize[0], gridSize[1]});

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

        fn getIndexFromGridPos(gridPos : vec2i) -> u32{

            let floatGridSize = vec2f(gridSize);
            let wrapped = vec2f(wrap(f32(gridPos.x), floatGridSize.x), wrap(f32(gridPos.y), floatGridSize.y));

            return u32(wrapped.x + wrapped.y * floatGridSize.x);
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
            
            particles[global_invocation_index].position = p.oldPosition + p.velocity * simulation.deltaTime;
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
        
            p.oldPosition = p.position;

            particles[global_invocation_index] = p;

            // let nextPos = particlesPast[global_invocation_index].position + 
            // particlesPast[global_invocation_index].velocity * simulation.deltaTime;
            
            // var screenWarpVec = vec2f(0,0);

            // let particleSizeTimes2 = globals.particleSize * 2;

            // if(nextPos.x > globals.canvasSize.x + globals.particleSize){
            //     screenWarpVec += vec2f(-globals.canvasSize.x - particleSizeTimes2, 0);
            // } else if (nextPos.x < - globals.particleSize) {
            //     screenWarpVec += vec2f(globals.canvasSize.x + particleSizeTimes2, 0);
            // }

            // if(nextPos.y > globals.canvasSize.y + globals.particleSize){
            //     screenWarpVec = vec2f(0, -globals.canvasSize.y - particleSizeTimes2);
            // } else if (nextPos.y < - globals.particleSize) {
            //     screenWarpVec = vec2f(0, globals.canvasSize.y + particleSizeTimes2);
            // }

            // //particles[global_invocation_index].gridCoords

            // let gridIndexN = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(0,1));
            // let gridIndexNE = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(1,1));
            // let gridIndexE = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(1,0));
            // let gridIndexSE = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(1,-1));
            // let gridIndexS = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(0,-1));
            // let gridIndexSW = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(-1,-1));
            // let gridIndexW = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(-1,0));
            // let gridIndexNW = getIndexFromGridPos(particles[global_invocation_index].gridCoords + vec2i(-1,1));

            // // let idN = atomicLoad(&grid[gridIndexN]);
            // // let idNE = atomicLoad(&grid[gridIndexNE]);
            // // let idE = atomicLoad(&grid[gridIndexE]);
            // // let idSE = atomicLoad(&grid[gridIndexSE]);
            // // let idS = atomicLoad(&grid[gridIndexSW]);
            // // let idSW = atomicLoad(&grid[gridIndexW]);
            // // let idW = atomicLoad(&grid[gridIndexW]);
            // // let idNW = atomicLoad(&grid[gridIndexNW]);
            // let idN = grid[gridIndexN];
            // let idNE = grid[gridIndexNE];
            // let idE = grid[gridIndexE];
            // let idSE = grid[gridIndexSE];
            // let idS = grid[gridIndexSW];
            // let idSW = grid[gridIndexW];
            // let idW = grid[gridIndexW];
            // let idNW = grid[gridIndexNW];

            // if(idN != 0 || idNE != 0 || idE != 0 || idSE != 0 || idS != 0 || idSW != 0 || idW != 0 || idNW != 0){
            //     particles[global_invocation_index].color = vec3f(1,0,0);
            // } else {
            //     particles[global_invocation_index].color = vec3f(0,1,0);
            // }


            // particles[global_invocation_index].position = nextPos + screenWarpVec;

        } 
        `,
      label,
      device
    );
  }
}
