import { describe, it, expect } from 'vitest';
import { AttentionModel } from './attention';

describe('AttentionModel', () => {
  it('initializes and computes tensor shapes correctly', () => {
    const model = new AttentionModel({ B: 2, T: 3, C: 4, D: 4 });

    expect(model.X.length).toBe(2);
    expect(model.X[0].length).toBe(3);
    expect(model.X[0][0].length).toBe(4);

    expect(model.Wq.length).toBe(4);
    expect(model.Wq[0].length).toBe(4);

    expect(model.Q.length).toBe(2);
    expect(model.Q[0].length).toBe(3);
    expect(model.Q[0][0].length).toBe(4);

    expect(model.S.length).toBe(2);
    expect(model.S[0].length).toBe(3);
    expect(model.S[0][0].length).toBe(3);

    expect(model.A.length).toBe(2);
    expect(model.A[0].length).toBe(3);
    expect(model.A[0][0].length).toBe(3);

    expect(model.O.length).toBe(2);
    expect(model.O[0].length).toBe(3);
    expect(model.O[0][0].length).toBe(4);
  });

  it('validates config parameters', () => {
    expect(() => new AttentionModel({ B: 0 })).toThrow();
    expect(() => new AttentionModel({ T: -1 })).toThrow();
    expect(() => new AttentionModel({ C: 0 })).toThrow();
    expect(() => new AttentionModel({ D: 0 })).toThrow();
  });

  it('computes Softmax probabilities that sum to 1 per row', () => {
    const model = new AttentionModel({ B: 1, T: 4, C: 4, D: 4 });

    for (let b = 0; b < model.config.B; b++) {
      for (let i = 0; i < model.config.T; i++) {
        const sumA = model.A[b][i].reduce((acc, val) => acc + val, 0);
        expect(sumA).toBeCloseTo(1.0, 5);
      }
    }
  });

  it('correctly provides Q dependencies', () => {
    const model = new AttentionModel({ B: 1, T: 2, C: 2, D: 2 });
    const deps = model.getQDependencies(0, 1, 0);

    expect(deps.type).toBe('Q');
    expect(deps.inputs.length).toBe(2);
    expect(deps.weights.length).toBe(2);
    expect(deps.result).toBe(model.Q[0][1][0]);
  });

  it('correctly provides S dependencies', () => {
    const model = new AttentionModel({ B: 1, T: 2, C: 2, D: 2 });
    const deps = model.getSDependencies(0, 0, 1);

    expect(deps.type).toBe('S');
    expect(deps.qRow.length).toBe(2);
    expect(deps.kRow.length).toBe(2);
    expect(deps.scale).toBe(Math.sqrt(2));
    expect(deps.result).toBeCloseTo(model.S[0][0][1], 5);
  });

  it('correctly provides O dependencies', () => {
    const model = new AttentionModel({ B: 1, T: 2, C: 2, D: 2 });
    const deps = model.getODependencies(0, 0, 1);

    expect(deps.type).toBe('O');
    expect(deps.aRow.length).toBe(2);
    expect(deps.vCol.length).toBe(2);
    expect(deps.result).toBeCloseTo(model.O[0][0][1], 5);
  });
});
