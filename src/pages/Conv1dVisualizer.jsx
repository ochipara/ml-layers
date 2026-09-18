import React, { useState, useMemo } from 'react';
import { Conv1dModel, getInputValue1d, getWeightValue1d } from '../core/conv1d';
import { getPaddingValue } from '../core/conv2d';
import { ConfigPanel1d } from '../components/conv1d/ConfigPanel1d';
import { MathBreakdown } from '../components/MathBreakdown';

function TensorRow({ title, channels, L, onHover, activeHighlights = [], renderCell }) {
  const channelsArray = Array.from({ length: channels }, (_, i) => i);

  return (
    <div className="flex flex-col border border-gray-300 rounded p-4 bg-white shadow-sm overflow-auto">
      <h3 className="font-bold mb-4 border-b pb-2">{title} ({channels}x{L})</h3>

      <div className="flex flex-col gap-4" onMouseLeave={() => onHover(null)}>
        {channelsArray.map(c => (
          <div key={c} className="flex flex-row gap-2 items-center">
             <span className="text-xs font-bold text-gray-500 w-16">Ch {c}</span>
             <div className="flex gap-1 bg-gray-100 p-1 rounded">
               {Array.from({ length: L }).map((_, l) => {
                 const isHighlighted = activeHighlights.some(hl => hl.c === c && hl.l === l);

                 let cellContent = null;
                 let extraClass = isHighlighted ? 'bg-yellow-300 border-yellow-500 font-bold' : 'bg-white border-gray-200';

                 if (renderCell) {
                   const rendered = renderCell(c, l, isHighlighted);
                   cellContent = rendered.content;
                   extraClass = rendered.className || extraClass;
                 }

                 return (
                   <div
                     key={l}
                     className={`border w-8 h-8 flex items-center justify-center text-xs cursor-pointer transition-colors shadow-sm ${extraClass}`}
                     onMouseEnter={() => onHover({ c, l })}
                   >
                     {cellContent}
                   </div>
                 )
               })}
             </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Conv1dVisualizer() {
  const [config, setConfig] = useState({
    L_in: 7,
    in_channels: 2, out_channels: 2,
    kernel_size: 3,
    stride: 1, padding: 1, dilation: 1, groups: 1,
    padding_mode: 'zeros', bias: true
  });

  const [error, setError] = useState(null);
  const [hoverState, setHoverState] = useState(null);

  const model = useMemo(() => {
    try {
      const m = new Conv1dModel(config);
      setError(null);
      return m;
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [config]);

  const outputShape = model ? model.getOutputShape() : null;

  const handleHoverOutput = (pos) => {
    if (!pos || !model) return setHoverState(null);
    const deps = model.getOutputDependencies(pos.c, pos.l);
    setHoverState({
      type: 'output',
      cout: pos.c, hout: 0, wout: pos.l,
      deps: {
          ...deps,
          inputs: deps.inputs.map(d => ({ ...d, h: 0, w: d.l })),
          weights: deps.weights.map(d => ({ ...d, kh: 0, kw: d.k }))
      }
    });
  };

  const handleHoverInput = (pos) => {
    if (!pos || !model) return setHoverState(null);
    const deps = model.getInputDependencies(pos.c, pos.l);
    setHoverState({
      type: 'input',
      cin: pos.c, hin: 0, win: pos.l,
      deps: deps.map(d => ({ ...d, hout: 0, wout: d.lout }))
    });
  };

  const handleHoverWeight = (pos) => {
    if (!pos || !model) return setHoverState(null);
    const deps = model.getWeightDependencies(pos.c, pos.cin_group, pos.l);
    setHoverState({
      type: 'weight',
      cout: pos.c, cin_group: pos.cin_group, kh: 0, kw: pos.l,
      deps: deps.map(d => ({ ...d, hout: 0, wout: d.lout, input: { ...d.input, h: 0, w: d.input.l } }))
    });
  };

  if (!model) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <ConfigPanel1d config={config} setConfig={setConfig} error={error} />
      </div>
    );
  }

  const p = config.padding;
  const L_pad = config.L_in + 2 * p;

  let inputHighlights = [];
  let weightHighlights = [];
  let outputHighlights = [];

  if (hoverState?.type === 'output') {
    outputHighlights = [{ c: hoverState.cout, l: hoverState.wout }];
    inputHighlights = hoverState.deps.inputs.map(d => ({ c: d.c, l: d.w + p }));
    weightHighlights = hoverState.deps.weights.map(d => ({ c: d.cout, cin_group: d.cin_group, l: d.kw }));
  } else if (hoverState?.type === 'input') {
    inputHighlights = [{ c: hoverState.cin, l: hoverState.win + p }];
    outputHighlights = hoverState.deps.map(d => ({ c: d.cout, l: d.wout }));
  } else if (hoverState?.type === 'weight') {
    weightHighlights = [{ c: hoverState.cout, cin_group: hoverState.cin_group, l: hoverState.kw }];
    outputHighlights = hoverState.deps.map(d => ({ c: d.cout, l: d.wout }));
    inputHighlights = hoverState.deps.map(d => ({ c: d.input.c, l: d.input.w + p }));
  }

  const cin_per_group = config.in_channels / config.groups;
  const weightChannels = config.out_channels * cin_per_group;

  const renderInputCell = (c, l, isHighlighted) => {
    const unpadded_l = l - p;
    const padInfo = getPaddingValue(unpadded_l, config.L_in, p, config.padding_mode);

    const isVirtual = !padInfo.valid || unpadded_l < 0 || unpadded_l >= config.L_in;

    let val = 0;
    if (!isVirtual) {
       val = getInputValue1d(c, unpadded_l);
    } else {
       if (config.padding_mode === 'zeros') {
          val = 0;
       } else {
          val = padInfo.valid ? getInputValue1d(c, padInfo.idx) : 0;
       }
    }

    let className = isHighlighted ? 'bg-yellow-300 border-yellow-500 font-bold' : (isVirtual ? 'bg-blue-50 border-blue-200 text-blue-400' : 'bg-white border-gray-200');
    return { content: val, className };
  };

  const renderWeightCell = (c_ui, l, isHighlighted) => {
    const cout = Math.floor(c_ui / cin_per_group);
    const cin_group = c_ui % cin_per_group;
    const val = getWeightValue1d(cout, cin_group, l);
    return { content: val, className: isHighlighted ? 'bg-orange-300 border-orange-500 font-bold' : 'bg-orange-50 border-orange-200' };
  };

  const renderOutputCell = (c, l, isHighlighted) => {
    const deps = model.getOutputDependencies(c, l);
    return { content: deps.result, className: isHighlighted ? 'bg-green-300 border-green-500 font-bold' : 'bg-green-50 border-green-200' };
  };

  return (
    <div className="p-8 max-w-7xl mx-auto flex flex-col gap-8">
      <ConfigPanel1d config={config} setConfig={setConfig} error={error} />

      <div className="flex flex-row gap-8 items-start">
        <TensorRow
          title="Padded Input X"
          channels={config.in_channels}
          L={L_pad}
          onHover={(pos) => {
             if (!pos) return handleHoverInput(null);
             handleHoverInput({ c: pos.c, l: pos.l - p });
          }}
          activeHighlights={inputHighlights}
          renderCell={renderInputCell}
        />

        <TensorRow
          title="Weights W (C_out x C_in_group)"
          channels={weightChannels}
          L={model.k}
          onHover={(pos) => {
             if (!pos) return handleHoverWeight(null);
             const cout = Math.floor(pos.c / cin_per_group);
             const cin_group = pos.c % cin_per_group;
             handleHoverWeight({ c: cout, cin_group, l: pos.l });
          }}
          activeHighlights={weightHighlights.map(hl => ({ c: hl.c * cin_per_group + hl.cin_group, l: hl.l }))}
          renderCell={renderWeightCell}
        />

        <TensorRow
          title="Output Y"
          channels={outputShape.C_out}
          L={outputShape.L_out}
          onHover={handleHoverOutput}
          activeHighlights={outputHighlights}
          renderCell={renderOutputCell}
        />
      </div>

      <MathBreakdown hoverData={hoverState} config={config} outShape={outputShape} />
    </div>
  );
}
