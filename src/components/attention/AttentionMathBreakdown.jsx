import React from 'react';

export function AttentionMathBreakdown({ hoverData, config }) {
  if (!hoverData) {
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 h-32 flex flex-col justify-center text-gray-500">
        <div className="text-center italic">
          Hover over any matrix cell (<i>X</i>, <i>W<sub>q</sub></i>, <i>W<sub>k</sub></i>, <i>W<sub>v</sub></i>, <i>Q</i>, <i>K</i>, <i>V</i>, <i>S</i>, <i>A</i>, <i>O</i>) to view exact dependencies and computation breakdowns.
        </div>
      </div>
    );
  }

  const { type, deps } = hoverData;

  if (type === 'X') {
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-blue-700">
          Input Tensor <i>X</i>[batch={hoverData.b}, time={hoverData.r}, channel={hoverData.c}] = {hoverData.val}
        </h3>
        <p className="text-sm bg-blue-50 p-2 rounded inline-block">
          Used to calculate Queries <i>Q</i>[{hoverData.r}, :], Keys <i>K</i>[{hoverData.r}, :], and Values <i>V</i>[{hoverData.r}, :].
        </p>
      </div>
    );
  }

  if (type === 'Wq' || type === 'Wk' || type === 'Wv') {
    const labelMap = {
      Wq: <span>Query Weight <i>W<sub>q</sub></i></span>,
      Wk: <span>Key Weight <i>W<sub>k</sub></i></span>,
      Wv: <span>Value Weight <i>W<sub>v</sub></i></span>
    };
    const colorMap = { Wq: 'text-purple-700 bg-purple-50', Wk: 'text-pink-700 bg-pink-50', Wv: 'text-emerald-700 bg-emerald-50' };
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className={`font-bold my-1 ${colorMap[type].split(' ')[0]}`}>
          {labelMap[type]}[channel={hoverData.r}, head_dim={hoverData.c}] = {hoverData.val}
        </h3>
        <p className={`text-sm p-2 rounded inline-block ${colorMap[type].split(' ')[1]}`}>
          Projects input channels to head dimension across all time steps <i>T</i>.
        </p>
      </div>
    );
  }

  if (type === 'Q' || type === 'K' || type === 'V') {
    const labelMap = {
      Q: <span>Query <i>Q</i></span>,
      K: <span>Key <i>K</i></span>,
      V: <span>Value <i>V</i></span>
    };
    const colorMap = { Q: 'text-purple-700 bg-purple-50', K: 'text-pink-700 bg-pink-50', V: 'text-emerald-700 bg-emerald-50' };
    const wSub = type === 'Q' ? 'q' : (type === 'K' ? 'k' : 'v');

    let equation = deps.inputs.map((inp, i) => {
      const w = deps.weights[i];
      return `(${inp.val} × ${w.val})`;
    }).join(' + ');

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className={`font-bold my-1 ${colorMap[type].split(' ')[0]}`}>
          {labelMap[type]}[t={deps.t}, d={deps.d}] = <i>X</i>[{deps.t}, :] · <i>W<sub>{wSub}</sub></i>[:, {deps.d}] = {deps.result}
        </h3>
        <div className={`font-mono text-sm break-words p-2 rounded ${colorMap[type].split(' ')[1]}`}>
          {equation} = <span className="font-bold">{deps.result}</span>
        </div>
      </div>
    );
  }

  if (type === 'S') {
    let equation = deps.qRow.map((q, idx) => {
      const k = deps.kRow[idx];
      return `(${q.val} × ${k.val})`;
    }).join(' + ');

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-amber-700">
          Raw Attention Score <i>S</i>[i={deps.i}, j={deps.j}] = (<i>Q</i>[{deps.i}, :] · <i>K</i>[{deps.j}, :]) / √{config.D}
        </h3>
        <div className="font-mono text-sm break-words bg-amber-50 p-2 rounded">
          [{equation}] / {deps.scale.toFixed(3)} = {deps.dot} / {deps.scale.toFixed(3)} = <span className="font-bold">{deps.result.toFixed(3)}</span>
        </div>
      </div>
    );
  }

  if (type === 'A') {
    const rowStr = deps.sRow.map((s, idx) => `S[${deps.i}, ${idx}] = ${s.val.toFixed(2)}`).join(', ');
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-indigo-700">
          Softmax Attention Weight <i>A</i>[i={deps.i}, j={deps.j}] = exp(<i>S</i>[{deps.i}, {deps.j}]) / ∑<sub>k</sub> exp(<i>S</i>[{deps.i}, k])
        </h3>
        <div className="font-mono text-sm break-words bg-indigo-50 p-2 rounded">
          Row scores: [{rowStr}] ⇒ Softmax Probability: <span className="font-bold">{(deps.result * 100).toFixed(1)}% ({deps.result.toFixed(4)})</span>
        </div>
      </div>
    );
  }

  if (type === 'O') {
    let equation = deps.aRow.map((a, idx) => {
      const v = deps.vCol[idx];
      return `(${a.val.toFixed(3)} × ${v.val})`;
    }).join(' + ');

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-teal-700">
          Output <i>O</i>[t={deps.i}, d={deps.d}] = <i>A</i>[{deps.i}, :] · <i>V</i>[:, {deps.d}]
        </h3>
        <div className="font-mono text-sm break-words bg-teal-50 p-2 rounded">
          {equation} = <span className="font-bold">{deps.result.toFixed(3)}</span>
        </div>
      </div>
    );
  }

  return null;
}
