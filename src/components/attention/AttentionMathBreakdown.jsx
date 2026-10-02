import React from 'react';

export function AttentionMathBreakdown({ hoverData, config }) {
  if (!hoverData) {
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 h-32 flex flex-col justify-center text-gray-500">
        <div className="text-center italic">
          Hover over any matrix cell ($X, W_q, W_k, W_v, Q, K, V, S, A, O$) to view exact dependencies and computation breakdowns.
        </div>
      </div>
    );
  }

  const { type, deps } = hoverData;

  if (type === 'X') {
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-blue-700">
          Input Tensor X[batch={hoverData.b}, time={hoverData.r}, channel={hoverData.c}] = {hoverData.val}
        </h3>
        <p className="text-sm bg-blue-50 p-2 rounded inline-block">
          Used to calculate Queries $Q[{hoverData.r}, :]$, Keys $K[{hoverData.r}, :]$, and Values $V[{hoverData.r}, :]$.
        </p>
      </div>
    );
  }

  if (type === 'Wq' || type === 'Wk' || type === 'Wv') {
    const nameMap = { Wq: 'Query Weight Wq', Wk: 'Key Weight Wk', Wv: 'Value Weight Wv' };
    const colorMap = { Wq: 'text-purple-700 bg-purple-50', Wk: 'text-pink-700 bg-pink-50', Wv: 'text-emerald-700 bg-emerald-50' };
    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className={`font-bold my-1 ${colorMap[type].split(' ')[0]}`}>
          {nameMap[type]}[channel={hoverData.r}, head_dim={hoverData.c}] = {hoverData.val}
        </h3>
        <p className={`text-sm p-2 rounded inline-block ${colorMap[type].split(' ')[1]}`}>
          Projects input channels to head dimension across all time steps $T$.
        </p>
      </div>
    );
  }

  if (type === 'Q' || type === 'K' || type === 'V') {
    const nameMap = { Q: 'Query Q', K: 'Key K', V: 'Value V' };
    const colorMap = { Q: 'text-purple-700 bg-purple-50', K: 'text-pink-700 bg-pink-50', V: 'text-emerald-700 bg-emerald-50' };
    const wName = type === 'Q' ? 'Wq' : (type === 'K' ? 'Wk' : 'Wv');

    let equation = deps.inputs.map((inp, i) => {
      const w = deps.weights[i];
      return `(${inp.val} * ${w.val})`;
    }).join(' + ');

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className={`font-bold my-1 ${colorMap[type].split(' ')[0]}`}>
          {nameMap[type]}[t={deps.t}, d={deps.d}] = X[{deps.t}, :] · {wName}[:, {deps.d}] = {deps.result}
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
      return `(${q.val} * ${k.val})`;
    }).join(' + ');

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-amber-700">
          Raw Attention Score S[i={deps.i}, j={deps.j}] = (Q[{deps.i}, :] · K[{deps.j}, :]) / √{config.D}
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
          Softmax Attention Weight A[i={deps.i}, j={deps.j}] = exp(S[{deps.i}, {deps.j}]) / ∑_k exp(S[{deps.i}, k])
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
      return `(${a.val.toFixed(3)} * ${v.val})`;
    }).join(' + ');

    return (
      <div className="bg-white border p-4 rounded shadow mt-4 overflow-auto">
        <h3 className="font-bold my-1 text-teal-700">
          Output O[t={deps.i}, d={deps.d}] = A[{deps.i}, :] · V[:, {deps.d}]
        </h3>
        <div className="font-mono text-sm break-words bg-teal-50 p-2 rounded">
          {equation} = <span className="font-bold">{deps.result.toFixed(3)}</span>
        </div>
      </div>
    );
  }

  return null;
}
