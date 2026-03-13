/**
 * main.js — Main control logic
 * Handles: state machine, keyboard events, polling scheduler, UI updates
 */

// ── DOM References ────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const DOM = {
  serverBadge:      $('server-status-badge'),
  btnToggleServer:  $('btn-toggle-server'),
  inputSavePath:    $('input-save-path'),
  inputDesc:        $('input-desc'),
  btnStartCapture:  $('btn-start-capture'),
  btnStopCapture:   $('btn-stop-capture'),
  lblRecording:     $('lbl-recording-status'),
  lblFrameCount:    $('lbl-frame-count'),
  lblSystemStatus:  $('lbl-system-status'),
  lblAdb:           $('lbl-adb-status'),
  logBox:           $('log-box'),
  bigStatusBox:     $('big-status-box'),
  bigStatusMain:    $('big-status-main'),
  bigStatusSub:     $('big-status-sub'),
  inputEpisode:     $('input-episode'),
  btnEpisodeDec:    $('btn-episode-dec'),
  btnEpisodeInc:    $('btn-episode-inc'),
};

// ── Server start timestamp (for restart detection) ────────────────────────
let _serverStartTime = null;

// ── State ────────────────────────────────────────────────────
let state = {
  serverConnected: false,
  captureRunning:  false,
  recordingState:  'waiting',   // waiting | countdown | recording
  isRecording:     false,
  isCountdown:     false,
  countdownValue:  0,
  episode:         0,
};

// ── UI Update Functions ──────────────────────────────────────────────

function setServerBadge(status) {
  const map = {
    connected:    { cls: 'connected',  text: 'Connected' },
    connecting:   { cls: 'connecting', text: 'Connecting...' },
    disconnected: { cls: '',           text: 'Not connected' },
    checking:     { cls: 'connecting', text: 'Checking...' },
    'not found':  { cls: '',           text: 'Server Not Found' },
  };
  const s = map[status] || map['disconnected'];
  DOM.serverBadge.className = `status-badge ${s.cls}`.trim();
  DOM.serverBadge.textContent = s.text;
}

function setBigStatus(main, sub = '', type = 'normal') {
  DOM.bigStatusBox.className = `big-status-box ${type}`.trim();
  DOM.bigStatusMain.textContent = main;
  DOM.bigStatusSub.textContent  = sub;
}

function addLog(entry) {
  const line = document.createElement('div');
  line.className = `log-line ${entry.level}`;
  line.innerHTML = `<span class="log-time">${entry.time}</span><span class="log-msg">${entry.message}</span>`;
  DOM.logBox.appendChild(line);
  DOM.logBox.scrollTop = DOM.logBox.scrollHeight;
  // Remove oldest entries when exceeding 500
  if (DOM.logBox.children.length > 500) {
    DOM.logBox.removeChild(DOM.logBox.firstChild);
  }
}

function applyFullStatus(s) {
  if (!s.ok) return;

  // Server restart detection: timestamp change indicates a new process, auto-refresh page
  if (s.server_start_time !== undefined) {
    if (_serverStartTime === null) {
      _serverStartTime = s.server_start_time;
    } else if (_serverStartTime !== s.server_start_time) {
      location.reload();
      return;
    }
  }

  // Server status — badge always follows server_status, not limited by boolean diff
  const newConnected = s.server_connected;
  setServerBadge(s.server_status || (newConnected ? 'connected' : 'disconnected'));
  if (newConnected !== state.serverConnected) {
    state.serverConnected = newConnected;
    DOM.btnToggleServer.textContent = newConnected ? '🔌 Disconnect Server' : '🔗 Connect to server';
    DOM.btnStartCapture.disabled = !newConnected;
    if (!newConnected) ImageDisplay.stop();
  }

  // Capture system
  state.captureRunning  = s.capture_running;
  state.recordingState  = s.recording_state;
  state.isRecording     = s.is_recording;
  state.isCountdown     = s.is_countdown;
  state.countdownValue  = s.countdown_value;

  // Episode
  if (s.episode !== state.episode) {
    state.episode = s.episode;
    DOM.inputEpisode.value = s.episode;
  }

  // Frame count & system status
  DOM.lblFrameCount.textContent = s.frame_count;
  DOM.lblSystemStatus.textContent = s.capture_running ? 'Running' : 'Stopped';
  DOM.lblSystemStatus.style.color = s.capture_running ? '#27ae60' : '#e74c3c';

  // ADB
  DOM.lblAdb.textContent = s.adb_connected ? 'Connected' : 'Not connected';
  DOM.lblAdb.style.color = s.adb_connected ? '#27ae60' : '#e74c3c';

  // Recording status label & big status
  updateRecordingUI();

  // Logs
  (s.logs || []).forEach(addLog);

  // Button enable states
  DOM.btnStopCapture.disabled   = !s.capture_running;
  DOM.btnStartCapture.disabled  = !s.server_connected || s.capture_running;
  DOM.btnToggleServer.disabled  = s.capture_running;
}

function updateRecordingUI() {
  const { captureRunning, recordingState, isCountdown, countdownValue, episode } = state;

  if (!captureRunning) {
    setBigStatus('Waiting for system start', '', 'normal');
    DOM.lblRecording.textContent = 'Waiting for start';
    DOM.lblRecording.style.color = '#e67e22';
  } else if (isCountdown) {
    setBigStatus(`⏰ ${countdownValue}`, 'Ready to start recording...', 'countdown');
    DOM.lblRecording.textContent = `Countdown: ${countdownValue}`;
    DOM.lblRecording.style.color = '#e74c3c';
  } else if (recordingState === 'recording') {
    setBigStatus(`🔴 Recording Episode ${episode}`, 'Press Space to stop recording', 'recording');
    DOM.lblRecording.textContent = `Recording Episode ${episode}...`;
    DOM.lblRecording.style.color = '#e74c3c';
  } else {
    setBigStatus('🎯 Press Space to record', 'Press D to discard', 'waiting');
    DOM.lblRecording.textContent = 'Waiting for keyboard input...';
    DOM.lblRecording.style.color = '#3498db';
  }
}

// ── Keyboard Events ────────────────────────────────────────────────

document.addEventListener('keydown', async e => {
  // Do not respond if focus is in an input field
  if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

  if (e.code === 'Space') {
    e.preventDefault();
    if (!state.captureRunning) return;
    if (state.recordingState === 'recording') {
      await handleStop();
    } else if (state.recordingState === 'waiting') {
      await handleStart();
    } else if (state.recordingState === 'countdown') {
      // Press again to cancel countdown
      await API.dropRecording();
    }
  }

  if (e.code === 'KeyD') {
    e.preventDefault();
    if (!state.captureRunning) return;
    await handleDrop();
  }

});

// ── Recording Actions ────────────────────────────────────────────────

async function handleStart() {
  const res = await API.startRecording();
  if (!res.ok) addLog({ time: now(), level: 'error', message: res.error });
}

async function handleStop() {
  const res = await API.stopRecording();
  if (!res.ok) {
    addLog({ time: now(), level: 'error', message: res.error });
    return;
  }
  if (!res.quality_ok) {
    setBigStatus('⛔ Quality Check Failed', 'Hands remained still too long, episode discarded', 'quality_error');
    setTimeout(() => setBigStatus('🎯 Press Space to record', 'Press D to discard', 'waiting'), 3000);
  } else {
    setBigStatus('✅ Episode Saved', `Next: Episode ${res.episode}`, 'success');
    setTimeout(() => setBigStatus('🎯 Press Space to record', 'Press D to discard', 'waiting'), 2000);
  }
  XYZChart.clear();
}

async function handleDrop() {
  const res = await API.dropRecording();
  if (!res.ok) {
    addLog({ time: now(), level: 'error', message: res.error });
    return;
  }
  if (res.dropped) {
    setBigStatus(`🗑️ Episode ${res.episode} discarded`, 'Recording canceled', 'warning');
    setTimeout(() => setBigStatus('🎯 Press Space to record', 'Press D to discard', 'waiting'), 3000);
  }
}

function now() {
  return new Date().toTimeString().slice(0, 8);
}

// ── Button Events ────────────────────────────────────────────────

DOM.btnToggleServer.addEventListener('click', async () => {
  DOM.btnToggleServer.disabled = true;
  setServerBadge('checking');
  const res = await API.toggleServer();
  DOM.btnToggleServer.disabled = false;
  if (!res.ok) addLog({ time: now(), level: 'error', message: res.error });
});

function setControlsLocked(locked) {
  DOM.btnStartCapture.disabled  = locked;
  DOM.btnStopCapture.disabled   = locked;
  DOM.btnToggleServer.disabled  = locked;
  DOM.inputSavePath.disabled    = locked;
  DOM.inputDesc.disabled        = locked;
}

DOM.btnStartCapture.addEventListener('click', async () => {
  const path = DOM.inputSavePath.value.trim();
  const desc = DOM.inputDesc.value.trim();
  setControlsLocked(true);
  const res = await API.startCapture(path, desc);
  setControlsLocked(false);
  if (!res.ok) {
    addLog({ time: now(), level: 'error', message: `Start failed: ${res.error}` });
    setBigStatus('❌ System Start Failed', 'Check the logs for detailed information', 'error');
  } else {
    setBigStatus('✅ System Ready', 'Press Space to start recording', 'success');
    setTimeout(() => setBigStatus('🎯 Press Space to record', 'Press D to discard', 'waiting'), 2000);
    ImageDisplay.start();
  }
});

DOM.btnStopCapture.addEventListener('click', async () => {
  setControlsLocked(true);
  const res = await API.stopCapture();
  setControlsLocked(false);
  ImageDisplay.stop();
  XYZChart.clear();
  setBigStatus('⏹️ System Stopped', 'Acquisition has ended', 'normal');
  if (!res.ok) addLog({ time: now(), level: 'error', message: res.error });
});

async function applyEpisode(n) {
  if (isNaN(n) || n < 0) return;
  const res = await API.setEpisode(n);
  if (res.ok) {
    DOM.inputEpisode.value = res.episode;
    addLog({ time: now(), level: 'record', message: `Episode set to ${res.episode}` });
  } else {
    addLog({ time: now(), level: 'error', message: `Set episode failed: ${res.error}` });
  }
}

DOM.inputEpisode.addEventListener('change', () => {
  applyEpisode(parseInt(DOM.inputEpisode.value, 10));
});

DOM.btnEpisodeInc.addEventListener('click', () => {
  const n = parseInt(DOM.inputEpisode.value, 10) || 0;
  applyEpisode(n + 1);
});

DOM.btnEpisodeDec.addEventListener('click', () => {
  const n = parseInt(DOM.inputEpisode.value, 10) || 0;
  applyEpisode(Math.max(0, n - 1));
});

// ── Polling Scheduler ────────────────────────────────────────────────

const _timerStatus = setInterval(async () => {
  try {
    const s = await API.getFullStatus();
    applyFullStatus(s);
  } catch (_) {}
}, 1000);

const _timerXyz = setInterval(async () => {
  if (!state.captureRunning) return;
  try {
    const res = await API.getRealtimeData();
    if (res.ok && res.data) XYZChart.update(res.data);
  } catch (_) {}
}, 100);

// Stop all polling before page unload to avoid broken pipe
window.addEventListener('beforeunload', () => {
  clearInterval(_timerStatus);
  clearInterval(_timerXyz);
  ImageDisplay.stop();
});

// ── Server Shutdown Event Listener ───────────────────────────────────
const _evtSource = new EventSource('/api/events');
_evtSource.onmessage = (e) => {
  if (e.data === 'shutdown') {
    _evtSource.close();
    clearInterval(_timerStatus);
    clearInterval(_timerXyz);
    ImageDisplay.stop();
    window.close();
    // window.close() may not work for non-script-opened tabs, fallback to showing a message
    document.body.innerHTML =
      '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;font-size:18px;color:#64748b;">Server has been shut down. Please close this tab manually.</div>';
  }
};

// ── Initialization ──────────────────────────────────────────────────

// Fetch initial status after page load
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(async () => {
    try {
      const s = await API.getFullStatus();
      applyFullStatus(s);
      if (s.server_connected) {
        setServerBadge('connected');
        ImageDisplay.setPlaceholder('Image server connected. Click Start Acquisition to display live feed.');
      }
    } catch (_) {}
  }, 500);
});
