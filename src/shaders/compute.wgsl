struct ParticleData{
    position: vec2f
}

@group(0) @binding(0) var<storage, read> lastFrame: ParticleData;
@group(0) @binding(1) var<storage, read_write> currentFrame: ParticleData;







