// headless Chromium with a fake microphone: a plain WebRTC peer (getUserMedia + RTCPeerConnection, offer POSTed to
// the Pipecat runner's /api/offer), records the bot audio and reports RTP stats
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const browser = await chromium.launch({ args: [
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${process.argv[2]}%noloop`, '--autoplay-policy=no-user-gesture-required',
    '--proxy-bypass-list=<-loopback>'] });
  const page = await (await browser.newContext({ permissions: ['microphone'] })).newPage();
  await page.goto('http://127.0.0.1:7860/client/');
  const setup = await page.evaluate(async () => {
    const t0 = performance.now();
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    const pc = new RTCPeerConnection(); window.__pc = pc; window.__chunks = [];
    pc.addTransceiver(mic.getAudioTracks()[0], { direction: 'sendrecv' });
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.createDataChannel('chat', { ordered: true });
    pc.addEventListener('track', (e) => {
      if (e.track.kind !== 'audio') return;
      const el = document.createElement('audio'); el.autoplay = true; el.srcObject = new MediaStream([e.track]); document.body.appendChild(el);
      const rec = new MediaRecorder(new MediaStream([e.track]), { mimeType: 'audio/webm;codecs=opus' });
      rec.ondataavailable = (d) => d.data.size && window.__chunks.push(d.data);
      rec.start(500); window.__rec = rec;
    });
    await pc.setLocalDescription(await pc.createOffer());
    await new Promise((r) => { if (pc.iceGatheringState === 'complete') r(); else pc.addEventListener('icegatheringstatechange', () => pc.iceGatheringState === 'complete' && r()); });
    const res = await fetch('/api/offer', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: 'offer' }) });
    const ans = await res.json();
    await pc.setRemoteDescription({ sdp: ans.sdp, type: ans.type });
    await new Promise((r) => { const c = () => pc.connectionState === 'connected' && r(); pc.addEventListener('connectionstatechange', c); c(); });
    return { ms_to_connected: Math.round(performance.now() - t0), answer_codecs: (ans.sdp.match(/a=rtpmap:\d+ [^\r\n]+/g) || []).slice(0, 4) };
  });
  console.log('setup', JSON.stringify(setup));
  const t0 = Date.now();
  for (let i = 0; i < 28; i++) {
    await page.waitForTimeout(1000);
    const s = await page.evaluate(async () => {
      const r = { state: window.__pc.connectionState };
      (await window.__pc.getStats()).forEach((v) => {
        if (v.type === 'inbound-rtp' && v.kind === 'audio') { r.inKB = Math.round(v.bytesReceived / 1000); r.energy = +(v.totalAudioEnergy || 0).toFixed(3); }
        if (v.type === 'outbound-rtp' && v.kind === 'audio') r.outKB = Math.round(v.bytesSent / 1000);
        if (v.type === 'candidate-pair' && v.nominated) r.rttMs = Math.round((v.currentRoundTripTime || 0) * 1000);
      });
      return r;
    });
    if (i % 2 == 1) console.log(`t=${((Date.now() - t0) / 1000).toFixed(0)}s`, JSON.stringify(s));
  }
  const b64 = await page.evaluate(async () => {
    window.__rec && window.__rec.stop(); await new Promise((r) => setTimeout(r, 800));
    const u = new Uint8Array(await new Blob(window.__chunks, { type: 'audio/webm' }).arrayBuffer());
    let s = ''; for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]); return btoa(s);
  });
  fs.writeFileSync(process.argv[3], Buffer.from(b64, 'base64'));
  console.log('saved bot audio:', Buffer.from(b64, 'base64').length, 'bytes');
  await browser.close();
})();
