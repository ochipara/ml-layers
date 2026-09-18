import React from 'react';

export function ConfigPanel1d({ config, setConfig, error }) {
  const handleChange = (key, val) => {
    let parsed = val;
    if (key !== 'padding_mode' && key !== 'bias') {
      parsed = parseInt(val, 10);
    }
    setConfig(prev => ({ ...prev, [key]: parsed }));
  };

  return (
    <div className="bg-gray-100 p-4 rounded shadow-sm border border-gray-300">
      <h2 className="text-xl font-bold mb-4">Configuration (Conv1d)</h2>
      {error && <div className="mb-4 text-red-600 font-semibold">{error}</div>}

      <div className="grid grid-cols-2 gap-4 text-sm">
        <label className="flex flex-col">
          <span>L_in</span>
          <input type="number" min="1" className="border p-1" value={config.L_in} onChange={(e) => handleChange('L_in', e.target.value)} />
        </label>

        <label className="flex flex-col">
          <span>in_channels</span>
          <input type="number" min="1" className="border p-1" value={config.in_channels} onChange={(e) => handleChange('in_channels', e.target.value)} />
        </label>
        <label className="flex flex-col">
          <span>out_channels</span>
          <input type="number" min="1" className="border p-1" value={config.out_channels} onChange={(e) => handleChange('out_channels', e.target.value)} />
        </label>

        <label className="flex flex-col">
          <span>kernel_size</span>
          <input type="number" className="border p-1" value={config.kernel_size} onChange={(e) => handleChange('kernel_size', e.target.value)} />
        </label>
        <label className="flex flex-col">
          <span>stride</span>
          <input type="number" className="border p-1" value={config.stride} onChange={(e) => handleChange('stride', e.target.value)} />
        </label>

        <label className="flex flex-col">
          <span>padding</span>
          <input type="number" className="border p-1" value={config.padding} onChange={(e) => handleChange('padding', e.target.value)} />
        </label>
        <label className="flex flex-col">
          <span>dilation</span>
          <input type="number" className="border p-1" value={config.dilation} onChange={(e) => handleChange('dilation', e.target.value)} />
        </label>

        <label className="flex flex-col">
          <span>groups</span>
          <input type="number" min="1" className="border p-1" value={config.groups} onChange={(e) => handleChange('groups', e.target.value)} />
        </label>

        <label className="flex flex-col">
          <span>padding_mode</span>
          <select className="border p-1" value={config.padding_mode} onChange={(e) => handleChange('padding_mode', e.target.value)}>
            <option value="zeros">zeros</option>
            <option value="reflect">reflect</option>
            <option value="replicate">replicate</option>
            <option value="circular">circular</option>
          </select>
        </label>

        <label className="flex items-center gap-2 mt-4 col-span-2">
          <input type="checkbox" checked={config.bias} onChange={(e) => setConfig(prev => ({...prev, bias: e.target.checked}))} />
          <span>bias</span>
        </label>
      </div>
    </div>
  );
}
