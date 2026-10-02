import React from 'react';

export function ConfigPanelAttention({ config, setConfig, selectedBatch, setSelectedBatch, error }) {
  const handleChange = (key, val) => {
    const parsed = Math.max(1, parseInt(val, 10) || 1);
    setConfig(prev => {
      const nextConfig = { ...prev, [key]: parsed };
      if (selectedBatch >= nextConfig.B) {
        setSelectedBatch(nextConfig.B - 1);
      }
      return nextConfig;
    });
  };

  return (
    <div className="bg-gray-100 p-4 rounded shadow-sm border border-gray-300">
      <h2 className="text-xl font-bold mb-4">Configuration (Attention Head)</h2>
      {error && <div className="mb-4 text-red-600 font-semibold">{error}</div>}

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm items-end">
        <label className="flex flex-col">
          <span className="font-semibold text-gray-700">Batch Size (B)</span>
          <input
            type="number"
            min="1"
            className="border bg-white p-1.5 rounded focus:outline-blue-500"
            value={config.B}
            onChange={(e) => handleChange('B', e.target.value)}
          />
        </label>

        <label className="flex flex-col">
          <span className="font-semibold text-gray-700">Sequence Length / Time (T)</span>
          <input
            type="number"
            min="1"
            className="border bg-white p-1.5 rounded focus:outline-blue-500"
            value={config.T}
            onChange={(e) => handleChange('T', e.target.value)}
          />
        </label>

        <label className="flex flex-col">
          <span className="font-semibold text-gray-700">Input Channels (C)</span>
          <input
            type="number"
            min="1"
            className="border bg-white p-1.5 rounded focus:outline-blue-500"
            value={config.C}
            onChange={(e) => handleChange('C', e.target.value)}
          />
        </label>

        <label className="flex flex-col">
          <span className="font-semibold text-gray-700">Head Dimension (D)</span>
          <input
            type="number"
            min="1"
            className="border bg-white p-1.5 rounded focus:outline-blue-500"
            value={config.D}
            onChange={(e) => handleChange('D', e.target.value)}
          />
        </label>

        <label className="flex flex-col">
          <span className="font-semibold text-blue-900">Active Batch (b)</span>
          <select
            className="border bg-blue-50 font-bold p-1.5 rounded focus:outline-blue-500 text-blue-900"
            value={selectedBatch}
            onChange={(e) => setSelectedBatch(parseInt(e.target.value, 10))}
          >
            {Array.from({ length: config.B }, (_, idx) => (
              <option key={idx} value={idx}>
                Batch {idx}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
