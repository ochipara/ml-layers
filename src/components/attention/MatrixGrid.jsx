import React from 'react';

export function MatrixGrid({
  title,
  rows,
  cols,
  rowPrefix = 'r',
  colPrefix = 'c',
  getValue,
  onHoverCell,
  getHighlightClass,
  formatValue
}) {
  return (
    <div className="flex flex-col border border-gray-300 rounded p-4 bg-white shadow-sm overflow-auto">
      <h3 className="font-bold mb-3 border-b pb-2 text-sm text-gray-800">
        {title} <span className="text-xs font-normal text-gray-500">({rows} × {cols})</span>
      </h3>

      <div className="flex flex-col gap-1" onMouseLeave={() => onHoverCell && onHoverCell(null)}>
        {/* Column indices header */}
        <div className="flex flex-row gap-1 items-center ml-12 mb-1">
          {Array.from({ length: cols }).map((_, colIdx) => (
            <div key={colIdx} className="w-10 text-center text-[10px] font-mono text-gray-400">
              {colPrefix}{colIdx}
            </div>
          ))}
        </div>

        {Array.from({ length: rows }).map((_, rowIdx) => (
          <div key={rowIdx} className="flex flex-row gap-1 items-center">
            <span className="text-[11px] font-bold font-mono text-gray-500 w-11 text-right pr-1">
              {rowPrefix}{rowIdx}
            </span>
            <div className="flex gap-1 bg-gray-50 p-1 rounded">
              {Array.from({ length: cols }).map((_, colIdx) => {
                const val = getValue(rowIdx, colIdx);
                const isHighlighted = getHighlightClass ? getHighlightClass(rowIdx, colIdx) : '';
                const displayVal = formatValue ? formatValue(val) : val;

                return (
                  <div
                    key={colIdx}
                    className={`border w-10 h-8 flex items-center justify-center text-xs cursor-pointer transition-colors shadow-sm rounded-sm font-mono ${isHighlighted || 'bg-white border-gray-200 hover:border-blue-400'}`}
                    onMouseEnter={() => onHoverCell && onHoverCell({ r: rowIdx, c: colIdx })}
                  >
                    {displayVal}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
