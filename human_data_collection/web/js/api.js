/**
 * api.js — Wraps all Python API calls
 * All methods return Promises with unified error handling.
 */

const API = {
  async _call(method, body = null) {
    const url = `/api/${method}`;
    const options = {
      method: body !== null ? 'POST' : 'GET',
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== null) {
      options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    return await response.json();
  },

  // ── Server Management ─────────────────────────────
  toggleServer()            { return this._call('toggle_server', {}); },
  connectToExistingServer() { return this._call('connect_to_existing_server', {}); },
  checkServerConnection()   { return this._call('check_server_connection', {}); },
  getServerStatus()         { return this._call('get_server_status'); },

  // ── Capture System ───────────────────────────────
  startCapture(path, desc)  { return this._call('start_capture', { path, description: desc }); },
  stopCapture()             { return this._call('stop_capture', {}); },
  browsePath()              { return this._call('browse_path'); },
  setEpisode(n)             { return this._call('set_episode', { value: n }); },

  // ── Recording Control ───────────────────────────────
  startRecording()          { return this._call('start_recording', {}); },
  stopRecording()           { return this._call('stop_recording', {}); },
  dropRecording()           { return this._call('drop_recording', {}); },

  // ── Real-time Data ───────────────────────────────
  getFullStatus()           { return this._call('get_full_status'); },
  getRealtimeData()         { return this._call('get_realtime_data'); },
  getImageFrame()           { return this._call('get_image_frame'); },

  // ── Lifecycle ───────────────────────────────
  shutdown()                { return this._call('shutdown', {}); },
};
