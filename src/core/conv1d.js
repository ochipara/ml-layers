import { getPaddingValue } from './conv2d';

export function getInputValue1d(c, l) {
  return ((c + l) % 2 === 0) ? 1 : -1;
}

export function getWeightValue1d(cout, cin_group, k) {
  return ((cout + cin_group + k) % 2 === 0) ? 1 : -1;
}

export function getBiasValue1d(cout) {
  return (cout % 2 === 0) ? 1 : -1;
}

export class Conv1dModel {
  constructor(config) {
    this.config = {
      stride: 1,
      padding: 0,
      dilation: 1,
      groups: 1,
      bias: true,
      padding_mode: 'zeros',
      ...config
    };

    this.validate();

    this.k = this.config.kernel_size;
    this.s = this.config.stride;
    this.p = this.config.padding;
    this.d = this.config.dilation;

    this.L_out = Math.floor((this.config.L_in + 2 * this.p - this.d * (this.k - 1) - 1) / this.s + 1);
  }

  validate() {
    const { in_channels, out_channels, groups, L_in, padding_mode, padding } = this.config;
    if (in_channels % groups !== 0) throw new Error("in_channels must be divisible by groups");
    if (out_channels % groups !== 0) throw new Error("out_channels must be divisible by groups");

    if (padding_mode === 'reflect' && padding >= L_in) {
      throw new Error("Padding must be less than the input dimension for reflect mode");
    }
  }

  getOutputShape() {
    return {
      C_out: this.config.out_channels,
      L_out: this.L_out
    };
  }

  getOutputDependencies(cout, lout) {
    const deps = {
      inputs: [],
      weights: [],
      bias: null,
      result: 0
    };

    const cin_per_group = this.config.in_channels / this.config.groups;
    const group_idx = Math.floor(cout / (this.config.out_channels / this.config.groups));
    const start_cin = group_idx * cin_per_group;

    let sum = 0;
    const l_in_start = lout * this.s - this.p;

    for (let c = 0; c < cin_per_group; c++) {
      const cin = start_cin + c;
      for (let k = 0; k < this.k; k++) {
        const l_in = l_in_start + k * this.d;
        const pad = getPaddingValue(l_in, this.config.L_in, this.p, this.config.padding_mode);

        let inputVal = 0;
        let isPadding = !pad.valid;

        if (pad.valid) {
          inputVal = getInputValue1d(cin, pad.idx);
        } else {
          inputVal = pad.value !== undefined ? pad.value : 0;
        }

        const weightVal = getWeightValue1d(cout, c, k);

        deps.inputs.push({
          c: cin,
          l: l_in, // unpadded
          val: inputVal,
          isPadding,
          validL: pad.valid ? pad.idx : null
        });

        deps.weights.push({
          cout,
          cin_group: c,
          k,
          val: weightVal
        });

        sum += inputVal * weightVal;
      }
    }

    if (this.config.bias) {
      const b = getBiasValue1d(cout);
      deps.bias = { cout, val: b };
      sum += b;
    }

    deps.result = sum;
    return deps;
  }

  getInputDependencies(cin, lin) {
    const outputs = [];
    const cin_per_group = this.config.in_channels / this.config.groups;
    const group_idx = Math.floor(cin / cin_per_group);

    const cout_per_group = this.config.out_channels / this.config.groups;
    const start_cout = group_idx * cout_per_group;

    for (let lout = 0; lout < this.L_out; lout++) {
      const l_in_start = lout * this.s - this.p;
      let contributes = false;

      for (let k = 0; k < this.k; k++) {
        const l_in = l_in_start + k * this.d;
        const pad = getPaddingValue(l_in, this.config.L_in, this.p, this.config.padding_mode);

        if (pad.valid && pad.idx === lin) {
          contributes = true;
        }
      }

      if (contributes) {
        for (let c = 0; c < cout_per_group; c++) {
          outputs.push({ cout: start_cout + c, lout });
        }
      }
    }

    return outputs;
  }

  getWeightDependencies(cout, cin_group, k) {
    const outputs = [];
    const cin_per_group = this.config.in_channels / this.config.groups;
    const group_idx = Math.floor(cout / (this.config.out_channels / this.config.groups));
    const start_cin = group_idx * cin_per_group;
    const cin = start_cin + cin_group;

    for (let lout = 0; lout < this.L_out; lout++) {
      const l_in_start = lout * this.s - this.p;
      const l_in = l_in_start + k * this.d;

      outputs.push({
        cout, lout,
        input: { c: cin, l: l_in }
      });
    }

    return outputs;
  }
}
