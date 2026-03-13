/**
 * image.js — Image frame polling display (30ms interval)
 */

const ImageDisplay = (() => {
  const img         = document.getElementById('camera-img');
  const placeholder = document.getElementById('image-placeholder');
  let   timer       = null;
  let   running     = false;

  function showFrame(b64) {
    img.src = `data:image/jpeg;base64,${b64}`;
    img.style.display = 'block';
    placeholder.style.display = 'none';
  }

  function showPlaceholder(text) {
    img.style.display = 'none';
    placeholder.textContent = text;
    placeholder.style.display = 'block';
  }

  async function tick() {
    if (!running) return;
    try {
      const res = await API.getImageFrame();
      if (res.ok && res.data) {
        showFrame(res.data);
      }
    } catch (_) { /* silently ignore */ }
    timer = setTimeout(tick, 33); // ~30 FPS
  }

  return {
    start() {
      if (running) return;
      running = true;
      tick();
    },
    stop() {
      running = false;
      if (timer) { clearTimeout(timer); timer = null; }
      showPlaceholder('Waiting to connect to the image server...');
    },
    setPlaceholder(text) {
      showPlaceholder(text);
    },
  };
})();
