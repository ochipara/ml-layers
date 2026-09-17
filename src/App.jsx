import React, { useState } from 'react';
import { Conv2dVisualizer } from './pages/Conv2dVisualizer';
import { Conv1dVisualizer } from './pages/Conv1dVisualizer';

function App() {
  const [activeComponent, setActiveComponent] = useState(null);

  if (!activeComponent) {
    return (
      <div className="min-h-screen bg-gray-50 text-gray-900 font-sans p-8 flex flex-col items-center justify-center">
        <h1 className="text-4xl font-extrabold tracking-tight mb-4 text-blue-900">ML Component Visualizer</h1>
        <p className="text-gray-600 mb-8 max-w-xl text-center text-lg">
          Interactive web-based visualizers for standard machine-learning components. Understand the exact relationship between inputs, parameters, weights, and outputs.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-2xl">
          <button
            onClick={() => setActiveComponent('conv2d')}
            className="bg-white border border-gray-200 hover:border-blue-400 hover:shadow-lg transition-all rounded-xl p-6 flex flex-col items-center text-center group"
          >
            <h2 className="text-2xl font-bold text-gray-800 group-hover:text-blue-600 mb-2">Conv2d</h2>
            <p className="text-sm text-gray-500">2D Spatial Convolution (e.g., images)</p>
          </button>

          <button
            onClick={() => setActiveComponent('conv1d')}
            className="bg-white border border-gray-200 hover:border-blue-400 hover:shadow-lg transition-all rounded-xl p-6 flex flex-col items-center text-center group"
          >
            <h2 className="text-2xl font-bold text-gray-800 group-hover:text-blue-600 mb-2">Conv1d</h2>
            <p className="text-sm text-gray-500">1D Spatial Convolution (e.g., sequences)</p>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      <header className="bg-white border-b px-8 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-blue-900">
            {activeComponent === 'conv2d' ? 'Conv2d' : 'Conv1d'} Visualizer
          </h1>
        </div>
        <button
          onClick={() => setActiveComponent(null)}
          className="text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 py-1 px-3 rounded transition-colors"
        >
          ← Back to Index
        </button>
      </header>

      <main className="pb-24">
        {activeComponent === 'conv2d' && <Conv2dVisualizer />}
        {activeComponent === 'conv1d' && <Conv1dVisualizer />}
      </main>
    </div>
  );
}

export default App;
