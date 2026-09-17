import React from 'react';

export function MathBreakdown({ hoverData, config, outShape }) {
  const getShapeFormula = () => {
    if (!config) return null;
    if (config.H_in !== undefined) {
      return `Output Dimensions: H_out = floor((${config.H_in} + 2*${config.padding} - ${config.dilation}*(${config.kernel_size}-1) - 1) / ${config.stride} + 1) = ${outShape?.H_out} | W_out = floor((${config.W_in} + 2*${config.padding} - ${config.dilation}*(${config.kernel_size}-1) - 1) / ${config.stride} + 1) = ${outShape?.W_out}`;
    } else {
       return `Output Dimension: L_out = floor((${config.L_in} + 2*${config.padding} - ${config.dilation}*(${config.kernel_size}-1) - 1) / ${config.stride} + 1) = ${outShape?.L_out}`;
    }
  };

  if (!hoverData) {
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 h-32 flex flex-col justify-center text-gray-500">
        <div className="mb-2 text-sm font-mono">{getShapeFormula()}</div>
        <div className="text-center italic">Hover over an output, input, or weight to see dependencies.</div>
      </div>
    );
  }

  if (hoverData.type === 'output') {
    const { deps } = hoverData;
    let equation = deps.inputs.map((d, i) => {
      const w = deps.weights[i];
      return `(${d.val} * ${w.val})`;
    }).join(' + ');

    if (deps.bias) {
      equation += ` + ${deps.bias.val} (bias)`;
    }

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <div className="mb-2 text-sm font-mono text-gray-500 pb-2 border-b">{getShapeFormula()}</div>
        <h3 className="font-bold my-2 text-green-700">
          Output Y[{hoverData.cout}, {hoverData.hout !== undefined ? hoverData.hout + ', ' : ''}{hoverData.wout}] = {deps.result}
        </h3>
        <div className="font-mono text-sm break-words bg-gray-50 p-2 rounded">
          {equation} = <span className="font-bold">{deps.result}</span>
        </div>
      </div>
    );
  }

  if (hoverData.type === 'input') {
    return (
       <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
         <div className="mb-2 text-sm font-mono text-gray-500 pb-2 border-b">{getShapeFormula()}</div>
         <h3 className="font-bold my-2 text-blue-700">
          Input X[{hoverData.cin}, {hoverData.hin !== undefined ? hoverData.hin + ', ' : ''}{hoverData.win}]
         </h3>
         <p className="text-sm bg-blue-50 p-2 rounded inline-block">Contributes to {hoverData.deps.length} output(s).</p>
       </div>
    );
  }

  if (hoverData.type === 'weight') {
    return (
       <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
         <div className="mb-2 text-sm font-mono text-gray-500 pb-2 border-b">{getShapeFormula()}</div>
         <h3 className="font-bold my-2 text-orange-700">
          Weight W[{hoverData.cout}, {hoverData.cin_group}, {hoverData.kh !== undefined ? hoverData.kh + ', ' : ''}{hoverData.kw}]
         </h3>
         <p className="text-sm bg-orange-50 p-2 rounded inline-block">Contributes to {hoverData.deps.length} output(s).</p>
       </div>
    );
  }

  return null;
}
