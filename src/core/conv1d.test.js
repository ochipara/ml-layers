import { describe, it, expect } from 'vitest';
import { Conv1dModel } from './conv1d';

describe('Conv1dModel', () => {
  it('computes output shape for basic config', () => {
    const model = new Conv1dModel({
      L_in: 5,
      in_channels: 1, out_channels: 1,
      kernel_size: 3,
      stride: 1, padding: 0, dilation: 1
    });
    expect(model.getOutputShape()).toEqual({ C_out: 1, L_out: 3 });
  });

  it('handles padding', () => {
    const model = new Conv1dModel({
      L_in: 5,
      in_channels: 1, out_channels: 1,
      kernel_size: 3,
      stride: 1, padding: 1, dilation: 1
    });
    expect(model.getOutputShape()).toEqual({ C_out: 1, L_out: 5 });
  });

  it('gets correct output dependencies', () => {
    const model = new Conv1dModel({
      L_in: 5,
      in_channels: 2, out_channels: 2,
      kernel_size: 3,
      stride: 1, padding: 1, dilation: 1, groups: 1, bias: false
    });
    // 2 in_channels, kernel 3 -> 6 inputs/weights per output element
    const deps = model.getOutputDependencies(0, 0);
    expect(deps.inputs.length).toBe(6);
    expect(deps.weights.length).toBe(6);

    // With padding 1, the first l is -1, unpadded, l_in_start = 0 * 1 - 1 = -1
    // for k=0 -> l_in = -1
    // for k=1 -> l_in = 0
    // for k=2 -> l_in = 1
    expect(deps.inputs.find(d => d.c === 0 && d.l === -1)).toBeDefined();
    expect(deps.inputs.find(d => d.c === 0 && d.l === 0)).toBeDefined();
    expect(deps.inputs.find(d => d.c === 1 && d.l === 1)).toBeDefined();
  });
});
