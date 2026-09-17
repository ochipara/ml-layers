import { describe, it, expect } from 'vitest';
import { Conv2dModel, getPaddingValue, normalizeTuple } from './conv2d';

describe('normalizeTuple', () => {
  it('normalizes numbers', () => {
    expect(normalizeTuple(3)).toEqual([3, 3]);
  });
  it('normalizes arrays', () => {
    expect(normalizeTuple([3])).toEqual([3, 3]);
    expect(normalizeTuple([3, 4])).toEqual([3, 4]);
  });
  it('throws on invalid', () => {
    expect(() => normalizeTuple([1, 2, 3])).toThrow();
  });
});

describe('getPaddingValue', () => {
  it('handles zeros', () => {
    expect(getPaddingValue(-1, 5, 1, 'zeros')).toEqual({ valid: false, value: 0 });
    expect(getPaddingValue(5, 5, 1, 'zeros')).toEqual({ valid: false, value: 0 });
  });
  it('handles replicate', () => {
    expect(getPaddingValue(-1, 5, 1, 'replicate')).toEqual({ valid: true, idx: 0 });
    expect(getPaddingValue(5, 5, 1, 'replicate')).toEqual({ valid: true, idx: 4 });
  });
  it('handles reflect', () => {
    expect(getPaddingValue(-1, 5, 1, 'reflect')).toEqual({ valid: true, idx: 1 });
    expect(getPaddingValue(-2, 5, 2, 'reflect')).toEqual({ valid: true, idx: 2 });
    expect(getPaddingValue(5, 5, 1, 'reflect')).toEqual({ valid: true, idx: 3 });
  });
  it('handles circular', () => {
    expect(getPaddingValue(-1, 5, 1, 'circular')).toEqual({ valid: true, idx: 4 });
    expect(getPaddingValue(5, 5, 1, 'circular')).toEqual({ valid: true, idx: 0 });
  });
});

describe('Conv2dModel', () => {
  it('computes output shape for basic config', () => {
    const model = new Conv2dModel({
      H_in: 5, W_in: 5,
      in_channels: 1, out_channels: 1,
      kernel_size: 3,
      stride: 1, padding: 0, dilation: 1
    });
    expect(model.getOutputShape()).toEqual({ C_out: 1, H_out: 3, W_out: 3 });
  });

  it('computes output shape with padding', () => {
    const model = new Conv2dModel({
      H_in: 5, W_in: 5,
      in_channels: 1, out_channels: 1,
      kernel_size: 3,
      stride: 1, padding: 1, dilation: 1
    });
    expect(model.getOutputShape()).toEqual({ C_out: 1, H_out: 5, W_out: 5 });
  });

  it('computes output shape with dilation', () => {
    const model = new Conv2dModel({
      H_in: 5, W_in: 5,
      in_channels: 1, out_channels: 1,
      kernel_size: 3,
      stride: 1, padding: 0, dilation: 2
    });
    // K = 3, D = 2 -> effective K = 5. (5 + 0 - 5) / 1 + 1 = 1
    expect(model.getOutputShape()).toEqual({ C_out: 1, H_out: 1, W_out: 1 });
  });

  it('validates groups', () => {
    expect(() => new Conv2dModel({
      H_in: 5, W_in: 5,
      in_channels: 4, out_channels: 6,
      groups: 2, kernel_size: 3
    })).not.toThrow();

    expect(() => new Conv2dModel({
      H_in: 5, W_in: 5,
      in_channels: 4, out_channels: 5,
      groups: 2, kernel_size: 3
    })).toThrow();
  });

  it('gets output dependencies with groups', () => {
    const model = new Conv2dModel({
      H_in: 5, W_in: 5,
      in_channels: 4, out_channels: 6,
      groups: 2, kernel_size: 3
    });
    const deps = model.getOutputDependencies(0, 0, 0); // cout = 0 belongs to group 0
    // group 0 has cin from 0 to 1 (4/2 = 2 channels per group)
    // kernel is 3x3, so 2 * 9 = 18 inputs/weights
    expect(deps.inputs.length).toBe(18);
    expect(deps.weights.length).toBe(18);
  });
});

describe('Dependency Mappings', () => {
  const model = new Conv2dModel({
    H_in: 3, W_in: 3,
    in_channels: 1, out_channels: 1,
    kernel_size: 2,
    stride: 1, padding: 1, dilation: 1, padding_mode: 'zeros'
  });
  // H_out = Math.floor((3 + 2 - 2) / 1) + 1 = 4

  it('input dependencies', () => {
    // Top left of unpadded input is (0, 0)
    const outDeps = model.getInputDependencies(0, 0, 0);
    // Since padding=1, the first kernel placement (top-left is at -1,-1 in unpadded coordinates)
    // When hout=0, wout=0, kernel covers [-1, 0] x [-1, 0]. It contains (0,0).
    // When hout=1, wout=0, kernel covers [0, 1] x [-1, 0]. It contains (0,0).
    // When hout=0, wout=1, kernel covers [-1, 0] x [0, 1]. It contains (0,0).
    // When hout=1, wout=1, kernel covers [0, 1] x [0, 1]. It contains (0,0).
    expect(outDeps.length).toBe(4);
  });
});

describe('Dependency Numerical Correctness', () => {
  it('correctly calculates single element with zeros padding', () => {
    // 3x3 input, 3x3 kernel, padding=1
    const model = new Conv2dModel({
      H_in: 3, W_in: 3,
      in_channels: 1, out_channels: 1,
      kernel_size: 3,
      stride: 1, padding: 1, dilation: 1, padding_mode: 'zeros', bias: false
    });

    // Test output at 0, 0
    // kernel covers [-1, 0, 1] x [-1, 0, 1]
    const deps = model.getOutputDependencies(0, 0, 0);
    expect(deps.inputs.length).toBe(9);

    let expectedSum = 0;
    // Hand calculate expectedSum
    for(let kh=0; kh<3; kh++) {
      for(let kw=0; kw<3; kw++) {
        const hin = -1 + kh;
        const win = -1 + kw;
        let inputVal = 0;
        if(hin >= 0 && hin < 3 && win >= 0 && win < 3) {
           inputVal = ((0 + hin + win) % 2 === 0) ? 1 : -1;
        }
        const weightVal = ((0 + 0 + kh + kw) % 2 === 0) ? 1 : -1;
        expectedSum += inputVal * weightVal;
      }
    }
    expect(deps.result).toBe(expectedSum);
  });
});
