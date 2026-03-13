/**
 * chart.js — Hand position real-time line chart (fixed 10s sliding window)
 *
 * X-axis fixed 0~10s, latest data on the right (x=10), flows left, auto-disappears past left edge
 */

const XYZChart = (() => {
  const canvas = document.getElementById('xyz-chart');

  const WINDOW = 10; // seconds

  const DATASETS = [
    { label: 'L-X', key: 'left_x',  color: '#ef4444' },
    { label: 'L-Y', key: 'left_y',  color: '#22c55e' },
    { label: 'L-Z', key: 'left_z',  color: '#3b82f6' },
    { label: 'R-X', key: 'right_x', color: '#f97316' },
    { label: 'R-Y', key: 'right_y', color: '#a855f7' },
    { label: 'R-Z', key: 'right_z', color: '#06b6d4' },
  ];

  // Local buffer: stores { t, left_x, left_y, ... } raw timestamp points
  const buf = [];

  const chart = new Chart(canvas, {
    type: 'line',
    data: {
      datasets: DATASETS.map(d => ({
        label:           d.label,
        data:            [],
        borderColor:     d.color,
        backgroundColor: d.color + '22',
        borderWidth:     1.5,
        pointRadius:     0,
        tension:         0.3,
      })),
    },
    options: {
      animation:           false,
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { boxWidth: 12, padding: 8, font: { size: 11 } },
        },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          type: 'linear',
          min:  0,
          max:  WINDOW,
          ticks: {
            maxTicksLimit: 6,
            font:  { size: 10 },
            color: '#94a3b8',
            callback: v => v.toFixed(0) + 's',
          },
          grid: { color: '#e2e8f0' },
        },
        y: {
          ticks: { font: { size: 10 }, color: '#94a3b8' },
          grid:  { color: '#e2e8f0' },
        },
      },
    },
  });

  function update(data) {
    if (!data) return;
    const ts = data.timestamps;
    if (!ts.length) return;

    // Append latest frame from backend to local buffer
    const tLatest = ts[ts.length - 1];
    const point = { t: tLatest };
    DATASETS.forEach(d => { point[d.key] = data[d.key][ts.length - 1]; });
    buf.push(point);

    // Remove points outside the window
    while (buf.length && tLatest - buf[0].t > WINDOW) buf.shift();

    // Map buffer to fixed X-axis: x = WINDOW - (tLatest - t), latest point x=10, oldest point x≈0
    DATASETS.forEach((d, i) => {
      chart.data.datasets[i].data = buf.map(p => ({
        x: WINDOW - (tLatest - p.t),
        y: p[d.key],
      }));
    });

    chart.update('none');
  }

  function clear() {
    buf.length = 0;
    chart.data.datasets.forEach(d => { d.data = []; });
    chart.update('none');
  }

  function init() {
    chart.resize();
  }

  return { init, update, clear };
})();
