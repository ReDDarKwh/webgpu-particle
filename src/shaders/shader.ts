import {
    EntryPoints,
    StructDefinitions,
    VariableDefinitions,
    makeShaderDataDefinitions,
  } from 'webgpu-utils';
import stringTemplate from '../stringTemplate';

export default abstract class Shader{

    private _module: GPUShaderModule;
    public get module(): GPUShaderModule {
        return this._module;
    }
    
    private _uniforms: VariableDefinitions;
    public get uniforms(): VariableDefinitions {
        return this._uniforms;
    }
    private _storages: VariableDefinitions;
    public get storages(): VariableDefinitions {
        return this._storages;
    }
    
    private _structs: StructDefinitions;
    public get structs(): StructDefinitions {
        return this._structs;
    }
    
    private _entryPoints: EntryPoints;
    public get entryPoints(): EntryPoints {
        return this._entryPoints;
    }

    constructor(code: string, label:string, device: GPUDevice) {
        const defs = makeShaderDataDefinitions(code);
        this._uniforms = defs.uniforms;
        this._storages = defs.storages;
        this._structs = defs.structs;
        this._entryPoints = defs.entryPoints;
        this._module = device.createShaderModule({
            label,
            code
        });

        console.log(code);
    }

}