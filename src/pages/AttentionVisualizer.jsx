import React, { useState, useMemo } from 'react';
import { AttentionModel } from '../core/attention';
import { ConfigPanelAttention } from '../components/attention/ConfigPanelAttention';
import { MatrixGrid } from '../components/attention/MatrixGrid';
import { AttentionMathBreakdown } from '../components/attention/AttentionMathBreakdown';

export function AttentionVisualizer() {
  const [config, setConfig] = useState({
    B: 1,
    T: 4,
    C: 4,
    D: 4
  });

  const [selectedBatch, setSelectedBatch] = useState(0);
  const [error, setError] = useState(null);
  const [hoverState, setHoverState] = useState(null);

  const model = useMemo(() => {
    try {
      const m = new AttentionModel(config);
      setError(null);
      return m;
    } catch (e) {
      setError(e.message);
      return null;
    }
  }, [config]);

  if (!model) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <ConfigPanelAttention
          config={config}
          setConfig={setConfig}
          selectedBatch={selectedBatch}
          setSelectedBatch={setSelectedBatch}
          error={error}
        />
      </div>
    );
  }

  const b = selectedBatch;

  // --- Hover Highlight Resolver ---
  const getHighlightClass = (matrixType, r, c) => {
    if (!hoverState) return '';

    const { type, b: hb } = hoverState;
    if (hb !== undefined && hb !== b && matrixType !== 'Wq' && matrixType !== 'Wk' && matrixType !== 'Wv') {
      return '';
    }

    // 1. Hover on X[r, c]
    if (type === 'X') {
      if (matrixType === 'X' && hoverState.r === r && hoverState.c === c) return 'bg-blue-300 border-blue-500 font-bold';
      if ((matrixType === 'Q' || matrixType === 'K' || matrixType === 'V') && r === hoverState.r) return 'bg-blue-100 border-blue-300';
      if ((matrixType === 'S' || matrixType === 'A') && (r === hoverState.r || c === hoverState.r)) return 'bg-blue-100 border-blue-300';
      if (matrixType === 'O' && r === hoverState.r) return 'bg-blue-100 border-blue-300';
    }

    // 2. Hover on Weights Wq, Wk, Wv
    if (type === 'Wq') {
      if (matrixType === 'Wq' && hoverState.r === r && hoverState.c === c) return 'bg-purple-300 border-purple-500 font-bold';
      if (matrixType === 'Q' && c === hoverState.c) return 'bg-purple-100 border-purple-300';
      if (matrixType === 'S' || matrixType === 'A' || matrixType === 'O') return 'bg-purple-50 border-purple-200';
    }
    if (type === 'Wk') {
      if (matrixType === 'Wk' && hoverState.r === r && hoverState.c === c) return 'bg-pink-300 border-pink-500 font-bold';
      if (matrixType === 'K' && c === hoverState.c) return 'bg-pink-100 border-pink-300';
      if (matrixType === 'S' || matrixType === 'A' || matrixType === 'O') return 'bg-pink-50 border-pink-200';
    }
    if (type === 'Wv') {
      if (matrixType === 'Wv' && hoverState.r === r && hoverState.c === c) return 'bg-emerald-300 border-emerald-500 font-bold';
      if (matrixType === 'V' && c === hoverState.c) return 'bg-emerald-100 border-emerald-300';
      if (matrixType === 'O' && c === hoverState.c) return 'bg-emerald-100 border-emerald-300';
    }

    // 3. Hover on Projections Q, K, V
    if (type === 'Q') {
      if (matrixType === 'Q' && hoverState.r === r && hoverState.c === c) return 'bg-purple-300 border-purple-500 font-bold';
      if (matrixType === 'X' && r === hoverState.r) return 'bg-blue-200 border-blue-400';
      if (matrixType === 'Wq' && c === hoverState.c) return 'bg-purple-200 border-purple-400';
      if ((matrixType === 'S' || matrixType === 'A' || matrixType === 'O') && r === hoverState.r) return 'bg-purple-100 border-purple-300';
    }

    if (type === 'K') {
      if (matrixType === 'K' && hoverState.r === r && hoverState.c === c) return 'bg-pink-300 border-pink-500 font-bold';
      if (matrixType === 'X' && r === hoverState.r) return 'bg-blue-200 border-blue-400';
      if (matrixType === 'Wk' && c === hoverState.c) return 'bg-pink-200 border-pink-400';
      if ((matrixType === 'S' || matrixType === 'A') && c === hoverState.r) return 'bg-pink-100 border-pink-300';
    }

    if (type === 'V') {
      if (matrixType === 'V' && hoverState.r === r && hoverState.c === c) return 'bg-emerald-300 border-emerald-500 font-bold';
      if (matrixType === 'X' && r === hoverState.r) return 'bg-blue-200 border-blue-400';
      if (matrixType === 'Wv' && c === hoverState.c) return 'bg-emerald-200 border-emerald-400';
      if (matrixType === 'O' && c === hoverState.c) return 'bg-emerald-100 border-emerald-300';
    }

    // 4. Hover on Score Matrix S
    if (type === 'S') {
      if (matrixType === 'S' && hoverState.r === r && hoverState.c === c) return 'bg-amber-300 border-amber-500 font-bold';
      if (matrixType === 'Q' && r === hoverState.r) return 'bg-purple-200 border-purple-400';
      if (matrixType === 'K' && r === hoverState.c) return 'bg-pink-200 border-pink-400';
      if (matrixType === 'A' && r === hoverState.r && c === hoverState.c) return 'bg-indigo-200 border-indigo-400';
      if (matrixType === 'O' && r === hoverState.r) return 'bg-teal-100 border-teal-300';
    }

    // 5. Hover on Attention Weight Matrix A
    if (type === 'A') {
      if (matrixType === 'A' && hoverState.r === r && hoverState.c === c) return 'bg-indigo-300 border-indigo-500 font-bold';
      if (matrixType === 'S' && r === hoverState.r) return 'bg-amber-200 border-amber-400';
      if (matrixType === 'O' && r === hoverState.r) return 'bg-teal-100 border-teal-300';
    }

    // 6. Hover on Output O
    if (type === 'O') {
      if (matrixType === 'O' && hoverState.r === r && hoverState.c === c) return 'bg-teal-300 border-teal-500 font-bold';
      if (matrixType === 'A' && r === hoverState.r) return 'bg-indigo-200 border-indigo-400';
      if (matrixType === 'V' && c === hoverState.c) return 'bg-emerald-200 border-emerald-400';
    }

    return '';
  };

  return (
    <div className="p-8 max-w-7xl mx-auto flex flex-col gap-8">
      <ConfigPanelAttention
        config={config}
        setConfig={setConfig}
        selectedBatch={selectedBatch}
        setSelectedBatch={setSelectedBatch}
        error={error}
      />

      {/* Section 1: Inputs & Weights */}
      <div className="flex flex-col gap-2">
        <h2 className="text-lg font-bold text-gray-700 border-b pb-1">1. Input X and Weight Projections (W_q, W_k, W_v)</h2>
        <div className="flex flex-row gap-6 overflow-x-auto pb-2 items-start">
          <MatrixGrid
            title={<span>Input X (Batch {b})</span>}
            rows={config.T}
            cols={config.C}
            rowPrefix="t"
            colPrefix="c"
            getValue={(r, c) => model.X[b][r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'X', b, r: pos.r, c: pos.c, val: model.X[b][pos.r][pos.c] }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('X', r, c)}
          />

          <MatrixGrid
            title={<span>Weight W<sub>q</sub></span>}
            rows={config.C}
            cols={config.D}
            rowPrefix="c"
            colPrefix="d"
            getValue={(r, c) => model.Wq[r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'Wq', r: pos.r, c: pos.c, val: model.Wq[pos.r][pos.c] }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('Wq', r, c)}
          />

          <MatrixGrid
            title={<span>Weight W<sub>k</sub></span>}
            rows={config.C}
            cols={config.D}
            rowPrefix="c"
            colPrefix="d"
            getValue={(r, c) => model.Wk[r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'Wk', r: pos.r, c: pos.c, val: model.Wk[pos.r][pos.c] }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('Wk', r, c)}
          />

          <MatrixGrid
            title={<span>Weight W<sub>v</sub></span>}
            rows={config.C}
            cols={config.D}
            rowPrefix="c"
            colPrefix="d"
            getValue={(r, c) => model.Wv[r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'Wv', r: pos.r, c: pos.c, val: model.Wv[pos.r][pos.c] }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('Wv', r, c)}
          />
        </div>
      </div>

      {/* Section 2: Q, K, V Matrices */}
      <div className="flex flex-col gap-2">
        <h2 className="text-lg font-bold text-gray-700 border-b pb-1">2. Projected Tensors: Query Q, Key K, Value V</h2>
        <div className="flex flex-row gap-6 overflow-x-auto pb-2 items-start">
          <MatrixGrid
            title={<span>Query Q = X · W<sub>q</sub> (Batch {b})</span>}
            rows={config.T}
            cols={config.D}
            rowPrefix="t"
            colPrefix="d"
            getValue={(r, c) => model.Q[b][r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'Q', b, r: pos.r, c: pos.c, deps: model.getQDependencies(b, pos.r, pos.c) }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('Q', r, c)}
          />

          <MatrixGrid
            title={<span>Key K = X · W<sub>k</sub> (Batch {b})</span>}
            rows={config.T}
            cols={config.D}
            rowPrefix="t"
            colPrefix="d"
            getValue={(r, c) => model.K[b][r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'K', b, r: pos.r, c: pos.c, deps: model.getKDependencies(b, pos.r, pos.c) }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('K', r, c)}
          />

          <MatrixGrid
            title={<span>Value V = X · W<sub>v</sub> (Batch {b})</span>}
            rows={config.T}
            cols={config.D}
            rowPrefix="t"
            colPrefix="d"
            getValue={(r, c) => model.V[b][r][c]}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'V', b, r: pos.r, c: pos.c, deps: model.getVDependencies(b, pos.r, pos.c) }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('V', r, c)}
          />
        </div>
      </div>

      {/* Section 3: Attention Scores, Softmax Weights, Output */}
      <div className="flex flex-col gap-2">
        <h2 className="text-lg font-bold text-gray-700 border-b pb-1">3. Attention Calculation (S, Softmax A, Output O)</h2>
        <div className="flex flex-row gap-6 overflow-x-auto pb-2 items-start">
          <MatrixGrid
            title={<span>Scores S = Q · K<sup>T</sup> / √D (Batch {b})</span>}
            rows={config.T}
            cols={config.T}
            rowPrefix="q_t"
            colPrefix="k_t"
            getValue={(r, c) => model.S[b][r][c]}
            formatValue={(v) => v.toFixed(2)}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'S', b, r: pos.r, c: pos.c, deps: model.getSDependencies(b, pos.r, pos.c) }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('S', r, c)}
          />

          <MatrixGrid
            title={<span>Attention Weights A = softmax(S) (Batch {b})</span>}
            rows={config.T}
            cols={config.T}
            rowPrefix="q_t"
            colPrefix="k_t"
            getValue={(r, c) => model.A[b][r][c]}
            formatValue={(v) => (v * 100).toFixed(0) + '%'}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'A', b, r: pos.r, c: pos.c, deps: model.getADependencies(b, pos.r, pos.c) }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('A', r, c)}
          />

          <MatrixGrid
            title={<span>Output O = A · V (Batch {b})</span>}
            rows={config.T}
            cols={config.D}
            rowPrefix="t"
            colPrefix="d"
            getValue={(r, c) => model.O[b][r][c]}
            formatValue={(v) => v.toFixed(2)}
            onHoverCell={(pos) => pos ? setHoverState({ type: 'O', b, r: pos.r, c: pos.c, deps: model.getODependencies(b, pos.r, pos.c) }) : setHoverState(null)}
            getHighlightClass={(r, c) => getHighlightClass('O', r, c)}
          />
        </div>
      </div>

      <AttentionMathBreakdown hoverData={hoverState} config={config} />
    </div>
  );
}
