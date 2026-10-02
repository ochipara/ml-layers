export function getInputValueAttention(b, t, c) {
  return ((b + t + c) % 2 === 0) ? 1 : -1;
}

export function getWeightValueAttention(type, c, d) {
  // type: 'q', 'k', 'v'
  const offset = type === 'q' ? 1 : (type === 'k' ? 2 : 3);
  return ((c + d + offset) % 2 === 0) ? 1 : -1;
}

export class AttentionModel {
  constructor(config) {
    this.config = {
      B: 1,
      T: 4,
      C: 4,
      D: 4,
      ...config
    };

    this.validate();
    this.computeTensors();
  }

  validate() {
    const { B, T, C, D } = this.config;
    if (B < 1) throw new Error("Batch size B must be at least 1");
    if (T < 1) throw new Error("Sequence length T must be at least 1");
    if (C < 1) throw new Error("Input channels C must be at least 1");
    if (D < 1) throw new Error("Head dimension D must be at least 1");
  }

  computeTensors() {
    const { B, T, C, D } = this.config;

    // 1. Input X [B, T, C]
    this.X = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, t) =>
        Array.from({ length: C }, (_, c) => getInputValueAttention(b, t, c))
      )
    );

    // 2. Weights Wq, Wk, Wv [C, D]
    this.Wq = Array.from({ length: C }, (_, c) =>
      Array.from({ length: D }, (_, d) => getWeightValueAttention('q', c, d))
    );
    this.Wk = Array.from({ length: C }, (_, c) =>
      Array.from({ length: D }, (_, d) => getWeightValueAttention('k', c, d))
    );
    this.Wv = Array.from({ length: C }, (_, c) =>
      Array.from({ length: D }, (_, d) => getWeightValueAttention('v', c, d))
    );

    // 3. Projections Q, K, V [B, T, D]
    this.Q = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, t) =>
        Array.from({ length: D }, (_, d) => {
          let sum = 0;
          for (let c = 0; c < C; c++) {
            sum += this.X[b][t][c] * this.Wq[c][d];
          }
          return sum;
        })
      )
    );

    this.K = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, t) =>
        Array.from({ length: D }, (_, d) => {
          let sum = 0;
          for (let c = 0; c < C; c++) {
            sum += this.X[b][t][c] * this.Wk[c][d];
          }
          return sum;
        })
      )
    );

    this.V = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, t) =>
        Array.from({ length: D }, (_, d) => {
          let sum = 0;
          for (let c = 0; c < C; c++) {
            sum += this.X[b][t][c] * this.Wv[c][d];
          }
          return sum;
        })
      )
    );

    // 4. Attention Scores S [B, T, T] = (Q * K^T) / sqrt(D)
    const scale = Math.sqrt(D);
    this.S = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, i) =>
        Array.from({ length: T }, (_, j) => {
          let dot = 0;
          for (let d = 0; d < D; d++) {
            dot += this.Q[b][i][d] * this.K[b][j][d];
          }
          return dot / scale;
        })
      )
    );

    // 5. Softmax Attention Weights A [B, T, T]
    this.A = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, i) => {
        const rowScores = this.S[b][i];
        const maxScore = Math.max(...rowScores); // numeric stability
        const exps = rowScores.map(s => Math.exp(s - maxScore));
        const sumExp = exps.reduce((acc, e) => acc + e, 0);
        return exps.map(e => e / sumExp);
      })
    );

    // 6. Output O [B, T, D] = A * V
    this.O = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, i) =>
        Array.from({ length: D }, (_, d) => {
          let sum = 0;
          for (let j = 0; j < T; j++) {
            sum += this.A[b][i][j] * this.V[b][j][d];
          }
          return sum;
        })
      )
    );
  }

  // --- Dependency functions for UI highlighting & math breakdown ---

  getQDependencies(b, t, d) {
    const { C } = this.config;
    const inputs = [];
    const weights = [];
    let sum = 0;

    for (let c = 0; c < C; c++) {
      const xVal = this.X[b][t][c];
      const wVal = this.Wq[c][d];
      inputs.push({ b, t, c, val: xVal });
      weights.push({ c, d, val: wVal });
      sum += xVal * wVal;
    }

    return {
      type: 'Q',
      b, t, d,
      inputs,
      weights,
      result: sum
    };
  }

  getKDependencies(b, t, d) {
    const { C } = this.config;
    const inputs = [];
    const weights = [];
    let sum = 0;

    for (let c = 0; c < C; c++) {
      const xVal = this.X[b][t][c];
      const wVal = this.Wk[c][d];
      inputs.push({ b, t, c, val: xVal });
      weights.push({ c, d, val: wVal });
      sum += xVal * wVal;
    }

    return {
      type: 'K',
      b, t, d,
      inputs,
      weights,
      result: sum
    };
  }

  getVDependencies(b, t, d) {
    const { C } = this.config;
    const inputs = [];
    const weights = [];
    let sum = 0;

    for (let c = 0; c < C; c++) {
      const xVal = this.X[b][t][c];
      const wVal = this.Wv[c][d];
      inputs.push({ b, t, c, val: xVal });
      weights.push({ c, d, val: wVal });
      sum += xVal * wVal;
    }

    return {
      type: 'V',
      b, t, d,
      inputs,
      weights,
      result: sum
    };
  }

  getSDependencies(b, i, j) {
    const { D } = this.config;
    const qRow = [];
    const kRow = [];
    let dot = 0;

    for (let d = 0; d < D; d++) {
      const qVal = this.Q[b][i][d];
      const kVal = this.K[b][j][d];
      qRow.push({ b, t: i, d, val: qVal });
      kRow.push({ b, t: j, d, val: kVal });
      dot += qVal * kVal;
    }

    const scale = Math.sqrt(D);
    const result = dot / scale;

    return {
      type: 'S',
      b, i, j,
      qRow,
      kRow,
      dot,
      scale,
      result
    };
  }

  getADependencies(b, i, j) {
    const { T } = this.config;
    const sRow = [];
    for (let k = 0; k < T; k++) {
      sRow.push({ b, i, j: k, val: this.S[b][i][k] });
    }

    return {
      type: 'A',
      b, i, j,
      sRow,
      result: this.A[b][i][j]
    };
  }

  getODependencies(b, i, d) {
    const { T } = this.config;
    const aRow = [];
    const vCol = [];
    let sum = 0;

    for (let j = 0; j < T; j++) {
      const aVal = this.A[b][i][j];
      const vVal = this.V[b][j][d];
      aRow.push({ b, i, j, val: aVal });
      vCol.push({ b, t: j, d, val: vVal });
      sum += aVal * vVal;
    }

    return {
      type: 'O',
      b, i, d,
      aRow,
      vCol,
      result: sum
    };
  }
}
