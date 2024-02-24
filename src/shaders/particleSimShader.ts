import Shader from "./shader";
import common from "./common.wgsl?raw";

export default class particleSimShader extends Shader {
  /**
   *
   */
  constructor(workgroupSize: number, label: string, device: GPUDevice) {
    super(
      /* wgsl */ `

        ${common}

        struct SimulationUniforms{
            deltaTime: f32,
            attractorPos: vec2f,
            isAttractorEnabled : u32,
            currentFrame:u32
        }

        struct StaticSimulationUniforms{
            tempOnHit: f32,
            cooldownRate: f32,
            attractorMass: f32,
            E:f32
        }

        const G = 0.5;
        const E = 0.7;
        const MAX_COL = 10;
        const maxAttractorForce = 100;
        const cellsOffsets = array(0, -1, 1);
        const overlapCorrectionForce = 0.5;

        @group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
        @group(1) @binding(1) var<storage, read_write> particleHeads: array<atomic<i32>>;
        @group(1) @binding(2) var<storage, read_write> particleLists: array<i32>;

        @group(2) @binding(0) var<uniform> simulation : SimulationUniforms;
        @group(2) @binding(1) var<uniform> ssu: StaticSimulationUniforms;
        @group(2) @binding(2) var<storage, read> attractors: array<vec3f>;

        fn getIndexFromGridPos(gridPos : vec2u) -> u32{

            return gridPos.x + gridPos.y * globals.gridSize.x;
        }

        fn applyCollision(
            global_invocation_index: u32
        ){
            var p = particles[global_invocation_index];

            if(p.collisionOtherIndex != -1){

                let o = particles[p.collisionOtherIndex];

                if(o.collisionOtherIndex == i32(global_invocation_index)){

                    let vCom = (p.mass * p.nextVel + o.mass * o.vel) / (p.mass + o.mass);
                    let v = (1 + ssu.E) * vCom - ssu.E * p.nextVel;
                    p.nextVel = v;
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
            p.temp = max(0, p.temp - ssu.cooldownRate * simulation.deltaTime);
            particles[global_invocation_index] = p;
            
            applyCollision(global_invocation_index);
        }

        fn attract(particlePos : vec2f, attractorPos : vec2f, mass : f32) -> vec2f{
            var force = vec2f();
            let pToA = attractorPos - particlePos;
            let distance = pow(pToA.x, 2) + pow(pToA.y, 2);
            force += normalize(pToA) * min(maxAttractorForce, G * (mass * ssu.attractorMass / distance));
            return force;
        }

        fn applyForces(pos : vec2f, mass : f32) -> vec2f{
            
            var force = vec2f();
            if(simulation.isAttractorEnabled == 1){
                force += attract(pos, simulation.attractorPos, mass);
            }

            for(var i: u32 = 1; i < arrayLength(&attractors); i++){
                let aPos = attractors[i];
                force += attract(pos, aPos.xy, mass * aPos.z);
            }

            return force;
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

            var newPos = p.nextPos + p.nextVel * simulation.deltaTime + p.acc * (simulation.deltaTime * simulation.deltaTime * 0.5);
            let newAcc = applyForces(newPos, p.mass);
            var newVel = p.nextVel + (p.acc + newAcc) * (simulation.deltaTime * 0.5);

            var screenBounce = vec2f(1,1);

            if(newPos.x > globals.canvasSize.x){
                screenBounce.x = -1;
            } else if (newPos.x < 0) {
                screenBounce.x = -1;
            }

            if(newPos.y > globals.canvasSize.y){
                screenBounce.y = -1;
            } else if (newPos.y < 0) {
                screenBounce.y = -1;
            }

            newPos = clamp(newPos, vec2f(), globals.canvasSize);
            newVel *= screenBounce;

            p.pos = newPos;
            p.nextPos = newPos;
            p.acc = newAcc;
            p.vel = newVel;
            p.nextVel = newVel;

            let gridCoords = vec2u(p.pos / globals.gridCellSizeInPixels);
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

            let r2 = globals.particleSize * 2;
            let r2WithPadding = r2 + 1;

            var p = particles[global_invocation_index];
            p.collisionOtherIndex = -1;

            let gridCoords = vec2i(p.pos / globals.gridCellSizeInPixels);

            var colNum = 0;
            var partnerFound = 0;

            for(var i : u32 = 0; i < 3; i++)
            {
                let y = p.cellsOffsetsY[(i + u32(random(f32(simulation.currentFrame) * p.mass * 0.34615462) * 10)) % 3];

                for(var j : u32 = 0; j < 3; j++)
                {

                    let x = p.cellsOffsetsX[(j + u32(random(f32(simulation.currentFrame) * p.mass * 0.7863435) * 23)) % 3];

                    if(colNum > MAX_COL){
                        break;
                    }

                    let gridCoordsWithOffset = gridCoords + vec2i(x,y);

                    if( gridCoordsWithOffset.x < 0 ||
                        gridCoordsWithOffset.x > i32(globals.gridSize.x) - 1 ||
                        gridCoordsWithOffset.y < 0 ||
                        gridCoordsWithOffset.y > i32(globals.gridSize.y) - 1)
                    {
                        continue;
                    }

                    let gridCoordsIndex = getIndexFromGridPos(vec2u(gridCoordsWithOffset));

                    var particleIndex = atomicLoad(&particleHeads[gridCoordsIndex]);

                    while(particleIndex >= 0 && colNum <= MAX_COL){

                        if(particleIndex != i32(global_invocation_index)){

                            let o = particles[particleIndex];
                            let d = p.pos - o.pos;
                            let l = length(d);

                            if(l < r2WithPadding){
                            
                                let normal = normalize(d);
                                let overlap = r2WithPadding - l;
                                
                                p.nextVel += normal * overlap * overlapCorrectionForce;

                                if(l < r2){
                                    p.temp = min(1, p.temp + ssu.tempOnHit);
                                    if(partnerFound == 0){
                                        p.collisionOtherIndex = particleIndex;
                                        p.nextPos += normal * (r2-l) * 0.5;
                                        partnerFound = 1;
                                    }
                                }

                                colNum ++;
                            }

                        }

                        particleIndex = particleLists[particleIndex];
                    }
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
