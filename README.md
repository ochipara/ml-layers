# ML Component Visualizer

An interactive, web-based visualizer for standard machine-learning components (Conv1d and Conv2d). This tool makes the mathematical relationship between a component's input, parameters, weights, and output visually understandable.

## Features

- **Conv2d and Conv1d Support:** Visualizes PyTorch-equivalent convolution logic.
- **Full Parameter Configuration:** Customize spatial dimensions, channels, kernel size, stride, padding, padding mode (zeros, reflect, replicate, circular), dilation, groups, and bias.
- **Interactive Dependencies:** Hover over inputs, weights, or outputs to see exactly which tensors and elements participate in the computation.
- **Virtual Padding:** Clearly distinguishes between original input cells and padded cells.
- **Math Breakdown:** Shows the exact numerical calculation for a hovered output element.

## Architecture

The project strictly separates the mathematical dependency logic from the visualization/UI layer:
- **Core Math Model (`src/core/`)**: Independently testable PyTorch-equivalent implementations (`Conv2dModel`, `Conv1dModel`) that expose pure dependency relationships without UI concerns.
- **UI & Visualization (`src/components/`, `src/pages/`)**: React components that consume the dependency models to render tensor grids and handle highlight states.

## Installation & Running

### Requirements
- Node.js (v18+ recommended)

### Quick Start

1. Install dependencies:
   `npm install`

2. Run the local development server:
   `npm run dev &`

3. Open your browser to the local server URL provided (usually `http://localhost:5173`).

### Running Tests

The mathematical dependency logic is comprehensively unit-tested using Vitest.

`npm test`

To run tests in watch mode:

`npm run test:watch`

## Stack

- **Vite** for fast build tooling
- **React** for declarative UI and state management
- **Tailwind CSS** for styling
- **Vitest** for unit testing
