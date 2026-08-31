const state = { candidateId: null, profile: null, analysis: null, socket: null, microphoneStream: null };
const $ = (id) => document.getElementById(id);
const api = (path, options) => fetch(path, options).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Request failed'); return data; });
function feedback(id, message, error = false) { const node = $(id); node.textContent = message; node.classList.toggle('error', error); }
function setPermissionNotice(message, type = 'info') {
  const banner = $('permission-banner');
  const text = $('permission-message');
  text.textContent = message;
  banner.hidden = false;
  banner.classList.remove('warning', 'error');
  if (type === 'warning') banner.classList.add('warning');
  if (type === 'error') banner.classList.add('error');
}
function hidePermissionNotice() { $('permission-banner').hidden = true; }
async function requestMeetingPermission() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setPermissionNotice('This browser cannot access the microphone. Use Chrome or Edge and allow microphone access to listen to the Google Meet conversation.', 'error');
    return false;
  }

  setPermissionNotice('Requesting microphone access so this page can listen to the Google Meet conversation...', 'warning');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    state.microphoneStream = stream;
    setPermissionNotice('Microphone access granted. The page is ready to listen to the Google Meet conversation.', 'info');
    return true;
  } catch (error) {
    state.microphoneStream = null;
    setPermissionNotice('Permission denied. Please allow microphone access to continue listening to the conversation.', 'error');
    return false;
  }
}
function stopMicrophoneStream() {
  if (state.microphoneStream) {
    state.microphoneStream.getTracks().forEach((track) => track.stop());
    state.microphoneStream = null;
  }
}
async function showDashboard() {
  if (!state.candidateId) {
    feedback('resume-feedback', 'Load the resume before starting the interview.', true);
    return;
  }

  const permissionGranted = await requestMeetingPermission();
  if (!permissionGranted) {
    $('transcript-status').textContent = 'Microphone permission required';
    return;
  }

  $('setup-view').hidden = true;
  $('dashboard-view').hidden = false;
  hidePermissionNotice();
  startInterview();
}
function addTranscript(speaker, text, question = false) {
  const line = document.createElement('div');
  const speakerLabel = speaker === 'candidate' ? 'Candidate' : 'Interviewer';
  line.className = `transcript-line${question ? ' question' : ''}`;
  line.innerHTML = `<span class="speaker">${speakerLabel}</span>${text}`;
  $('transcript').appendChild(line);
  $('transcript').scrollTop = $('transcript').scrollHeight;
}
function renderAnswer(answer) { $('answer-empty').hidden = true; $('answer-content').hidden = false; $('answer-text').textContent = answer.suggested_answer; $('key-points').innerHTML = (answer.key_points || []).map((item) => `<li>${item}</li>`).join(''); $('followups').innerHTML = (answer.likely_followups || []).map((item) => `<li>${item}</li>`).join(''); $('answer-meta').textContent = `${answer.confidence} confidence · ${answer.direct_experience ? 'direct evidence found' : 'transferable approach'}`; }
async function loadResume() { try { const data = await api('/api/resume/current'); state.candidateId = data.candidate_id; state.profile = data.profile; $('profile-strip').hidden = false; $('profile-name').textContent = data.profile.name || 'Candidate profile'; $('profile-role').textContent = data.profile.current_role || 'Local resume loaded'; $('profile-avatar').textContent = (data.profile.name || 'C').slice(0,1).toUpperCase(); $('launch-button').disabled = false; feedback('resume-feedback', 'Local resume loaded and ready.'); } catch (error) { feedback('resume-feedback', error.message, true); } }
async function analyzeJob() { const text = $('job-description').value.trim(); if (!text) return feedback('job-feedback', 'Paste a job description first.', true); feedback('job-feedback', 'Analyzing role fit...'); try { state.analysis = await api('/api/job/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ job_description:text, candidate_id:state.candidateId }) }); feedback('job-feedback', `${state.analysis.job_title} analyzed.`); $('launch-button').disabled = !state.candidateId; } catch (error) { feedback('job-feedback', error.message, true); } }
async function refreshUsage() { try { const usage = await api('/api/usage'); $('usage-label').textContent = `${usage.daily_requests || 0} daily requests · ${usage.daily_tokens || 0} tokens`; } catch (_) { $('usage-label').textContent = 'Usage unavailable'; } }
function startInterview() { const protocol = location.protocol === 'https:' ? 'wss' : 'ws'; const socket = new WebSocket(`${protocol}://${location.host}/api/interview/ws/${crypto.randomUUID()}`); state.socket = socket; socket.onopen = () => { $('status-dot').classList.add('online'); $('connection-label').textContent = 'Connected'; $('listen-pill').textContent = 'LIVE'; $('listen-pill').classList.add('live'); $('transcript-status').textContent = 'Listening to the conversation'; socket.send(JSON.stringify({ candidate_id:state.candidateId, job_title:state.analysis?.job_title || 'Not provided', key_requirements:state.analysis?.required_skills || [] })); }; socket.onmessage = (event) => { const data = JSON.parse(event.data); if (data.type === 'ready') $('transcript-status').textContent = 'Listening for transcript'; if (data.type === 'transcript') { addTranscript('interviewer', data.text, data.is_question); if (data.answer) renderAnswer(data.answer); } if (data.error) $('transcript-status').textContent = data.error; }; socket.onclose = () => { $('status-dot').classList.remove('online'); $('connection-label').textContent = 'Disconnected'; $('listen-pill').textContent = 'IDLE'; $('listen-pill').classList.remove('live'); }; refreshUsage(); }
function sendTranscript() { const text = $('transcript-input').value.trim(); if (!text || !state.socket || state.socket.readyState !== WebSocket.OPEN) return; state.socket.send(JSON.stringify({ speaker:'interviewer', text })); $('transcript-input').value = ''; }
$('analyze-button').addEventListener('click', analyzeJob); $('launch-button').addEventListener('click', showDashboard); $('send-button').addEventListener('click', sendTranscript); $('transcript-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') sendTranscript(); }); $('end-button').addEventListener('click', () => { state.socket?.close(); stopMicrophoneStream(); $('dashboard-view').hidden = true; $('setup-view').hidden = false; }); loadResume();
fetch('/health').then((response) => response.json()).then((data) => { $('model-label').textContent = data.config?.model || 'offline'; }).catch(() => {});