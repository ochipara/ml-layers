export function normalizeTuple(val, dim = 2) {
  if (typeof val === 'number') {
    return Array(dim).fill(val);
  }
  if (Array.isArray(val)) {
    if (val.length === 1 && dim === 2) return [val[0], val[0]];
    if (val.length === dim) return val;
  }
  throw new Error("Invalid tuple format");
}

export function getPaddingValue(idx, inDim, pad, mode) {
  // idx is the coordinate in the logically unpadded space.
  // if idx >= 0 && idx < inDim, it's valid.
  // else it falls in the padding region.
  if (idx >= 0 && idx < inDim) return { valid: true, idx };

  if (mode === 'zeros') return { valid: false, value: 0 };

  if (mode === 'replicate') {
    return { valid: true, idx: Math.max(0, Math.min(inDim - 1, idx)) };
  }

  if (mode === 'reflect') {
    let r = idx;
    if (r < 0) {
      r = -r;
    } else if (r >= inDim) {
      r = 2 * inDim - 2 - r;
    }
    return { valid: true, idx: r };
  }

  if (mode === 'circular') {
    let r = idx % inDim;
    if (r < 0) r += inDim;
    return { valid: true, idx: r };
  }

  throw new Error(`Unknown padding mode: ${mode}`);
}

export function getInputValue(c, h, w) {
  return ((c + h + w) % 2 === 0) ? 1 : -1;
}

export function getWeightValue(cout, cin_group, h, w) {
  return ((cout + cin_group + h + w) % 2 === 0) ? 1 : -1;
}

export function getBiasValue(cout) {
  return (cout % 2 === 0) ? 1 : -1;
}

export class Conv2dModel {
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

    const [kh, kw] = normalizeTuple(this.config.kernel_size);
    const [sh, sw] = normalizeTuple(this.config.stride);
    const [ph, pw] = normalizeTuple(this.config.padding);
    const [dh, dw] = normalizeTuple(this.config.dilation);

    this.kh = kh; this.kw = kw;
    this.sh = sh; this.sw = sw;
    this.ph = ph; this.pw = pw;
    this.dh = dh; this.dw = dw;

    this.H_out = Math.floor((this.config.H_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
    this.W_out = Math.floor((this.config.W_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
  }

  validate() {
    const { in_channels, out_channels, groups, H_in, W_in, padding_mode, padding } = this.config;
    if (in_channels % groups !== 0) throw new Error("in_channels must be divisible by groups");
    if (out_channels % groups !== 0) throw new Error("out_channels must be divisible by groups");

    const [ph, pw] = normalizeTuple(padding);
    if (padding_mode === 'reflect') {
      if (ph >= H_in || pw >= W_in) {
        throw new Error("Padding must be less than the input dimension for reflect mode");
      }
    }
    // We'll add more validations if needed.
  }

  getOutputShape() {
    return {
      C_out: this.config.out_channels,
      H_out: this.H_out,
      W_out: this.W_out
    };
  }

  // Returns { inputs: [{c, h, w, val, isPadding}], weights: [{cout, cin_group, kh, kw, val}], bias: {cout, val}, result }
  getOutputDependencies(cout, hout, wout) {
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

    const h_in_start = hout * this.sh - this.ph;
    const w_in_start = wout * this.sw - this.pw;

    for (let c = 0; c < cin_per_group; c++) {
      const cin = start_cin + c;
      for (let kh = 0; kh < this.kh; kh++) {
        for (let kw = 0; kw < this.kw; kw++) {
          const h_in = h_in_start + kh * this.dh;
          const w_in = w_in_start + kw * this.dw;

          const padH = getPaddingValue(h_in, this.config.H_in, this.ph, this.config.padding_mode);
          const padW = getPaddingValue(w_in, this.config.W_in, this.pw, this.config.padding_mode);

          let inputVal = 0;
          let isPadding = false;

          if (padH.valid && padW.valid) {
            inputVal = getInputValue(cin, padH.idx, padW.idx);
          } else if (!padH.valid && !padW.valid) {
             // Both zero padding
             inputVal = 0;
             isPadding = true;
          } else if (!padH.valid) {
             inputVal = padH.value !== undefined ? padH.value : 0;
             isPadding = true;
          } else if (!padW.valid) {
             inputVal = padW.value !== undefined ? padW.value : 0;
             isPadding = true;
          }

          if (!padH.valid || !padW.valid) {
              isPadding = true;
          }

          const weightVal = getWeightValue(cout, c, kh, kw);

          deps.inputs.push({
            c: cin,
            h: h_in, // coordinate in unpadded/virtual space
            w: w_in,
            val: inputVal,
            isPadding,
            validH: padH.valid ? padH.idx : null,
            validW: padW.valid ? padW.idx : null
          });

          deps.weights.push({
            cout,
            cin_group: c,
            kh,
            kw,
            val: weightVal
          });

          sum += inputVal * weightVal;
        }
      }
    }

    if (this.config.bias) {
      const b = getBiasValue(cout);
      deps.bias = { cout, val: b };
      sum += b;
    }

    deps.result = sum;
    return deps;
  }

  // Find all output positions a specific input element contributes to
  // Returns [{ cout, hout, wout }]
  getInputDependencies(cin, hin, win) {
    const outputs = [];
    const cin_per_group = this.config.in_channels / this.config.groups;
    const group_idx = Math.floor(cin / cin_per_group);

    const cout_per_group = this.config.out_channels / this.config.groups;
    const start_cout = group_idx * cout_per_group;

    for (let hout = 0; hout < this.H_out; hout++) {
      for (let wout = 0; wout < this.W_out; wout++) {
        const h_in_start = hout * this.sh - this.ph;
        const w_in_start = wout * this.sw - this.pw;

        let contributes = false;

        for (let kh = 0; kh < this.kh; kh++) {
          for (let kw = 0; kw < this.kw; kw++) {
            const h_in = h_in_start + kh * this.dh;
            const w_in = w_in_start + kw * this.dw;

            const padH = getPaddingValue(h_in, this.config.H_in, this.ph, this.config.padding_mode);
            const padW = getPaddingValue(w_in, this.config.W_in, this.pw, this.config.padding_mode);

            if (padH.valid && padW.valid && padH.idx === hin && padW.idx === win) {
              contributes = true;
            }
          }
        }

        if (contributes) {
          for (let c = 0; c < cout_per_group; c++) {
            outputs.push({ cout: start_cout + c, hout, wout });
          }
        }
      }
    }

    return outputs;
  }

  // Find outputs that a specific weight contributes to
  // Returns [{ cout, hout, wout, input: {c, h, w} }]
  getWeightDependencies(cout, cin_group, kh, kw) {
    const outputs = [];
    const cin_per_group = this.config.in_channels / this.config.groups;
    const group_idx = Math.floor(cout / (this.config.out_channels / this.config.groups));
    const start_cin = group_idx * cin_per_group;
    const cin = start_cin + cin_group;

    for (let hout = 0; hout < this.H_out; hout++) {
      for (let wout = 0; wout < this.W_out; wout++) {
        const h_in_start = hout * this.sh - this.ph;
        const w_in_start = wout * this.sw - this.pw;

        const h_in = h_in_start + kh * this.dh;
        const w_in = w_in_start + kw * this.dw;

        outputs.push({
          cout, hout, wout,
          input: { c: cin, h: h_in, w: w_in }
        });
      }
    }

    return outputs;
  }
}
