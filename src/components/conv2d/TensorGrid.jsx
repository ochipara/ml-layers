import React from 'react';

export function TensorGrid({ title, channels, H, W, getValue, onHover, activeHighlights = [], renderCell }) {
  const channelsArray = Array.from({ length: channels }, (_, i) => i);

  return (
    <div className="flex flex-col border border-gray-300 rounded p-4 bg-white shadow-sm overflow-auto">
      <h3 className="font-bold mb-4 border-b pb-2">{title} ({channels}x{H}x{W})</h3>

      <div className="flex flex-col gap-6" onMouseLeave={() => onHover(null)}>
        {channelsArray.map(c => (
          <div key={c} className="flex flex-col gap-1 items-start">
             <span className="text-xs font-bold text-gray-500 mb-1">Channel {c}</span>
             <div
               className="grid gap-1 bg-gray-100 p-1 rounded"
               style={{ gridTemplateColumns: `repeat(${W}, minmax(0, 1fr))` }}
             >
                {Array.from({ length: H }).map((_, h) =>
                   Array.from({ length: W }).map((_, w) => {
                     const val = getValue ? getValue(c, h, w) : null;
                     const isHighlighted = activeHighlights.some(hl => hl.c === c && hl.h === h && hl.w === w);

                     let cellContent = val;
                     let extraClass = isHighlighted ? 'bg-yellow-300 border-yellow-500 font-bold' : 'bg-white border-gray-200';

                     if (renderCell) {
                       const rendered = renderCell(c, h, w, isHighlighted);
                       cellContent = rendered.content;
                       extraClass = rendered.className || extraClass;
                     }

                     return (
                       <div
                         key={`${h}-${w}`}
                         className={`border w-8 h-8 flex items-center justify-center text-xs cursor-pointer transition-colors shadow-sm ${extraClass}`}
                         onMouseEnter={() => onHover({ c, h, w })}
                       >
                         {cellContent}
                       </div>
                     )
                   })
                )}
             </div>
          </div>
        ))}
      </div>
    </div>
  );
}
