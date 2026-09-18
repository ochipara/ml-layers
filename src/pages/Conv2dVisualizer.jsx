import React, { useState, useMemo } from 'react';
import { Conv2dModel, getInputValue, getWeightValue, normalizeTuple, getPaddingValue } from '../core/conv2d';
import { ConfigPanel } from '../components/conv2d/ConfigPanel';
import { TensorGrid } from '../components/conv2d/TensorGrid';
import { MathBreakdown } from '../components/MathBreakdown';

export function Conv2dVisualizer() {
  const [config, setConfig] = useState({
    H_in: 5, W_in: 5,
    in_channels: 2, out_channels: 2,
    kernel_size: 3,
    stride: 1, padding: 1, dilation: 1, groups: 1,
    padding_mode: 'zeros', bias: true
  });

  const [error, setError] = useState(null);
  const [hoverState, setHoverState] = useState(null); // { type: 'input'|'weight'|'output', data: {...}, deps: [...] }

  const model = useMemo(() => {
    try {
      const m = new Conv2dModel(config);
      setError(null);
      return m;
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [config]);

  const outputShape = model ? model.getOutputShape() : null;

  // Handle Hover Output
  const handleHoverOutput = (pos) => {
    if (!pos || !model) {
      setHoverState(null);
      return;
    }
    const deps = model.getOutputDependencies(pos.c, pos.h, pos.w);
    setHoverState({
      type: 'output',
      cout: pos.c, hout: pos.h, wout: pos.w,
      deps
    });
  };

  const handleHoverInput = (pos) => {
    if (!pos || !model) {
      setHoverState(null);
      return;
    }
    const deps = model.getInputDependencies(pos.c, pos.h, pos.w);
    setHoverState({
      type: 'input',
      cin: pos.c, hin: pos.h, win: pos.w,
      deps
    });
  };

  const handleHoverWeight = (pos) => {
    if (!pos || !model) {
      setHoverState(null);
      return;
    }
    const deps = model.getWeightDependencies(pos.c, pos.cin_group, pos.h, pos.w);
    setHoverState({
      type: 'weight',
      cout: pos.c, cin_group: pos.cin_group, kh: pos.h, kw: pos.w,
      deps
    });
  };

  if (!model) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <ConfigPanel config={config} setConfig={setConfig} error={error} />
      </div>
    );
  }

  const [ph, pw] = normalizeTuple(config.padding);
  const H_pad = config.H_in + 2 * ph;
  const W_pad = config.W_in + 2 * pw;

  // Derive Highlights
  let inputHighlights = [];
  let weightHighlights = [];
  let outputHighlights = [];

  if (hoverState?.type === 'output') {
    outputHighlights = [{ c: hoverState.cout, h: hoverState.hout, w: hoverState.wout }];
    // Input highlights are in virtual padded space coordinates
    // We map back to unpadded if valid, or just highlight the padded region
    inputHighlights = hoverState.deps.inputs.map(d => {
       // d.h and d.w are in logically unpadded space.
       // The UI displays the PADDED tensor.
       // UI coordinate (h_ui, w_ui) maps to padded space.
       return { c: d.c, h: d.h + ph, w: d.w + pw };
    });
    weightHighlights = hoverState.deps.weights.map(d => ({ c: d.cout, cin_group: d.cin_group, h: d.kh, w: d.kw }));
  } else if (hoverState?.type === 'input') {
    inputHighlights = [{ c: hoverState.cin, h: hoverState.hin + ph, w: hoverState.win + pw }];
    outputHighlights = hoverState.deps.map(d => ({ c: d.cout, h: d.hout, w: d.wout }));
  } else if (hoverState?.type === 'weight') {
    weightHighlights = [{ c: hoverState.cout, cin_group: hoverState.cin_group, h: hoverState.kh, w: hoverState.kw }];
    outputHighlights = hoverState.deps.map(d => ({ c: d.cout, h: d.hout, w: d.wout }));
    inputHighlights = hoverState.deps.map(d => ({ c: d.input.c, h: d.input.h + ph, w: d.input.w + pw }));
  }

  // Weight rendering needs cin_group tabs rather than absolute cin.
  // Actually, weights are W[C_out, C_in/groups, K_h, K_w]
  // We can treat the "channels" for the weight tensor as (C_out * (C_in/groups)).
  // We will flatten it for the UI: c_ui = cout * (Cin/groups) + cin_group.
  const cin_per_group = config.in_channels / config.groups;
  const weightChannels = config.out_channels * cin_per_group;

  const renderInputCell = (c, h, w, isHighlighted) => {
    const unpadded_h = h - ph;
    const unpadded_w = w - pw;
    const padInfoH = getPaddingValue(unpadded_h, config.H_in, ph, config.padding_mode);
    const padInfoW = getPaddingValue(unpadded_w, config.W_in, pw, config.padding_mode);

    const isVirtual = !padInfoH.valid || !padInfoW.valid || unpadded_h < 0 || unpadded_h >= config.H_in || unpadded_w < 0 || unpadded_w >= config.W_in;

    let val = 0;
    if (!isVirtual) {
       val = getInputValue(c, unpadded_h, unpadded_w);
    } else {
       if (config.padding_mode === 'zeros') {
          val = 0;
       } else {
          const h_mapped = padInfoH.valid ? padInfoH.idx : 0;
          const w_mapped = padInfoW.valid ? padInfoW.idx : 0;
          if (padInfoH.valid && padInfoW.valid) val = getInputValue(c, h_mapped, w_mapped);
          else val = 0; // fallback just in case
       }
    }

    let className = isHighlighted ? 'bg-yellow-300 border-yellow-500 font-bold' : (isVirtual ? 'bg-blue-50 border-blue-200 text-blue-400' : 'bg-white border-gray-200');

    return { content: val, className };
  };

  const renderWeightCell = (c_ui, h, w, isHighlighted) => {
    const cout = Math.floor(c_ui / cin_per_group);
    const cin_group = c_ui % cin_per_group;
    const val = getWeightValue(cout, cin_group, h, w);
    return { content: val, className: isHighlighted ? 'bg-orange-300 border-orange-500 font-bold' : 'bg-orange-50 border-orange-200' };
  };

  const renderOutputCell = (c, h, w, isHighlighted) => {
    // Ideally we compute this to show the real output value
    const deps = model.getOutputDependencies(c, h, w);
    return { content: deps.result, className: isHighlighted ? 'bg-green-300 border-green-500 font-bold' : 'bg-green-50 border-green-200' };
  };

  return (
    <div className="p-8 max-w-7xl mx-auto flex flex-col gap-8">
      <ConfigPanel config={config} setConfig={setConfig} error={error} />

      <div className="flex flex-row gap-8 items-start">
        <TensorGrid
          title="Padded Input X"
          channels={config.in_channels}
          H={H_pad} W={W_pad}
          onHover={(pos) => {
             if (!pos) return handleHoverInput(null);
             // map back to unpadded coordinates for the logical model
             handleHoverInput({ c: pos.c, h: pos.h - ph, w: pos.w - pw });
          }}
          activeHighlights={inputHighlights}
          renderCell={renderInputCell}
        />

        <TensorGrid
          title="Weights W (C_out x C_in_group)"
          channels={weightChannels}
          H={model.kh} W={model.kw}
          onHover={(pos) => {
             if (!pos) return handleHoverWeight(null);
             const cout = Math.floor(pos.c / cin_per_group);
             const cin_group = pos.c % cin_per_group;
             handleHoverWeight({ c: cout, cin_group, h: pos.h, w: pos.w });
          }}
          activeHighlights={weightHighlights.map(hl => ({ c: hl.c * cin_per_group + hl.cin_group, h: hl.h, w: hl.w }))}
          renderCell={renderWeightCell}
        />

        <TensorGrid
          title="Output Y"
          channels={outputShape.C_out}
          H={outputShape.H_out} W={outputShape.W_out}
          onHover={handleHoverOutput}
          activeHighlights={outputHighlights}
          renderCell={renderOutputCell}
        />
      </div>

      <MathBreakdown hoverData={hoverState} config={config} outShape={outputShape} />
    </div>
  );
}
