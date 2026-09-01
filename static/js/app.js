const state = {
    activeTab: 'corkboard',
    privacyMode: 'BALANCED',
    customThreshold: 0.70,
    isRecording: false,
    recognition: null,
    audioContext: null,
    analyser: null,
    mediaStream: null,
    visualizerAnimId: null,
    voiceMeetingId: null,
    currentResults: null,
    activeFilter: 'all',
    selectedAudioFile: null,
    currentTheme: 'dark'
};
const SAMPLES = {
    1: `**Arjun:** Hey Meera, let's start the project sync. Can you provide an update on the dashboard?
**Meera:** Good morning! The main dashboard is functional. I still need to finish the filtering and clean up a few UI issues.
**Arjun:** Did you watch the match yesterday?
**Meera:** Yes, Real Madrid played amazingly! But speaking of work, I fixed the login issue on Friday.
**Arjun:** Excellent. What about the presentation for the upcoming review?
**Meera:** I think we should include the performance comparison rather than just showing final numbers.
**Arjun:** I agree. And maybe a small graph for response time. I'll handle the introduction and project overview, then you can explain the technical part.
**Meera:** Sounds good! The installation documentation is done. I will complete the API details and troubleshooting section by tomorrow.
**Arjun:** Great. First, finish the dashboard filters. Second, complete the documentation. Third, update the presentation.
**Meera:** And I'll push the authentication fix to the testing branch today.
**Arjun:** Perfect. I'll ask QA to start testing it. Send me the presentation outline after the meeting.
**Meera:** Will do. See you at lunch!`,
    2: `**Alex:** Morning Sarah, did you get coffee yet?
**Sarah:** Yes, finally awake! Without coffee I'm basically running on Windows 98.
**Alex:** Classic. How was your weekend?
**Sarah:** We went shopping for furniture and spent three hours deciding where to eat!
**Alex:** That's the most realistic weekend plan ever.
**Sarah:** Exactly! Coming back to our prototype, the sensor integration is completed.
**Alex:** Great. What is the power consumption?
**Sarah:** It consumes 80 milliamps at 3.3V during transmission. We decided to use ESP8266 as the micro-controller.
**Alex:** Decision approved. Priya will coordinate with the hardware vendor for component sourcing.
**Sarah:** I will write the test benchmark suite and evaluation metrics before Wednesday.
**Alex:** Action item: Sarah will push the firmware changes to the repository by end of day.`,
    3: `**Prof. Sharma:** Good afternoon team. Let's review the final semester project milestones.
**Rohan:** Good afternoon Professor. Our primary objective is to complete the dataset preprocessing by Friday.
**Ananya:** I have implemented the multi-class classifier and evaluated the baseline accuracy at 88 percent.
**Prof. Sharma:** Very good. We should evaluate using standard ROUGE and BLEU metrics rather than simple accuracy alone.
**Rohan:** Agreed. I will run the ROUGE-1, ROUGE-2, and ROUGE-L benchmark on the generated summaries.
**Ananya:** I will finalize the project report and prepare the slide deck by Thursday.
**Prof. Sharma:** Perfect. Let's schedule the final mock presentation for next Monday at 10 AM.`
};
function initThemeToggle() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const sunIcon = document.getElementById('theme-icon-sun');
    const moonIcon = document.getElementById('theme-icon-moon');
    const themeLabel = document.getElementById('theme-label');
    const savedTheme = localStorage.getItem('meetlytic-theme') || 'dark';
    setTheme(savedTheme);
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const newTheme = state.currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
        });
    }
    function setTheme(theme) {
        state.currentTheme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('meetlytic-theme', theme);
        if (theme === 'dark') {
            if (sunIcon) sunIcon.style.display = 'block';
            if (moonIcon) moonIcon.style.display = 'none';
            if (themeLabel) themeLabel.textContent = 'Light Mode';
        } else {
            if (sunIcon) sunIcon.style.display = 'none';
            if (moonIcon) moonIcon.style.display = 'block';
            if (themeLabel) themeLabel.textContent = 'Dark Mode';
        }
    }
}
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    const targetBtn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
    const targetContent = document.getElementById(`tab-${tabName}`);
    if (targetBtn && targetContent) {
        targetBtn.classList.add('active');
        targetContent.classList.add('active');
        state.activeTab = tabName;
        if (tabName === 'history') loadMeetings();
        if (tabName === 'evaluation') runEvaluationMetrics();
    }
}
function initSubtabs() {
    document.querySelectorAll('.subtab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.subtab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            const targetId = btn.dataset.subtab;
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add('active');
        });
    });
}
function loadSampleTranscript(sampleNum) {
    switchTab('input');
    const subtabBtn = document.querySelector('.subtab-btn[data-subtab="transcript-subtab"]');
    if (subtabBtn) subtabBtn.click();
    const textarea = document.getElementById('transcript-input');
    if (textarea && SAMPLES[sampleNum]) {
        textarea.value = SAMPLES[sampleNum].trim();
        showToast(`Loaded Preset Sample ${sampleNum}`, 'success');
    }
}
function clearTranscript() {
    const textarea = document.getElementById('transcript-input');
    if (textarea) textarea.value = '';
}
async function analyzeTranscript() {
    const text = document.getElementById('transcript-input').value.trim();
    if (!text) {
        showToast('Please enter or paste transcript text', 'error');
        return;
    }
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('processing-loading');
    analyzeBtn.disabled = true;
    loading.style.display = 'block';
    document.getElementById('loading-text').textContent = 'Extracting conversational features and synthesizing sticky notes...';
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                privacy_mode: state.privacyMode
            })
        });
        const data = await response.json();
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        state.currentResults = data;
        renderCorkboardStickyNotes(data.sticky_notes);
        renderClassificationAudit(data);
        renderWordWeightsHeatmap(data);
        showToast(`Generated ${data.sticky_notes.length} sticky notes from ${data.total_sentences} turns`, 'success');
        switchTab('corkboard');
    } catch (e) {
        console.error('Analysis error:', e);
        showToast('Analysis failed. Please try again.', 'error');
    } finally {
        analyzeBtn.disabled = false;
        loading.style.display = 'none';
    }
}
function initAudioDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio-file-input');
    const uploadBtn = document.getElementById('upload-audio-btn');
    const fileInfo = document.getElementById('selected-file-info');
    if (!dropZone || !fileInput) return;
    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleSelectedAudio(files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleSelectedAudio(e.target.files[0]);
    });
    function handleSelectedAudio(file) {
        state.selectedAudioFile = file;
        fileInfo.style.display = 'inline-block';
        fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        uploadBtn.disabled = false;
    }
    uploadBtn.addEventListener('click', async () => {
        if (!state.selectedAudioFile) return;
        const loading = document.getElementById('processing-loading');
        loading.style.display = 'block';
        document.getElementById('loading-text').textContent = 'Transcribing audio speech and extracting acoustic stress weightage...';
        uploadBtn.disabled = true;
        const formData = new FormData();
        formData.append('audio', state.selectedAudioFile);
        formData.append('privacy_mode', state.privacyMode);
        try {
            const response = await fetch('/api/upload-audio', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }
            state.currentResults = data;
            renderCorkboardStickyNotes(data.sticky_notes);
            renderClassificationAudit(data);
            renderAudioWaveform(data);
            renderWordWeightsHeatmap(data);
            showToast(`Audio Transcribed: ${data.sticky_notes.length} sticky notes pinned.`, 'success');
            switchTab('corkboard');
        } catch (e) {
            console.error('Audio upload failed:', e);
            showToast('Audio transcription failed.', 'error');
        } finally {
            loading.style.display = 'none';
            uploadBtn.disabled = false;
        }
    });
}
function renderAudioWaveform(data) {
    const container = document.getElementById('audio-waveform-container');
    const barsContainer = document.getElementById('waveform-bars');
    const meta = document.getElementById('waveform-meta');
    if (!container || !barsContainer) return;
    container.style.display = 'block';
    const profile = data.acoustic_profile || {};
    const duration = data.duration_sec || 0;
    meta.textContent = `Duration: ${duration}s | Peak RMS Energy: ${profile.peak_energy || 0.8} | Stress Ratio: ${((profile.stress_ratio || 0.2)*100).toFixed(0)}%`;
    barsContainer.innerHTML = '';
    const samples = profile.energy_samples && profile.energy_samples.length > 0
        ? profile.energy_samples
        : Array.from({length: 36}, () => Math.random() * 0.8 + 0.1);
    samples.forEach((val) => {
        const bar = document.createElement('div');
        bar.className = `waveform-bar ${val > 0.65 ? 'bar-stressed' : ''}`;
        const heightPct = Math.max(8, Math.min(100, val * 100));
        bar.style.height = `${heightPct}%`;
        barsContainer.appendChild(bar);
    });
}
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const micStatus = document.getElementById('mic-status');
    const micBtn = document.getElementById('mic-btn');
    if (!SpeechRecognition) {
        if (micStatus) micStatus.textContent = 'Live voice input not supported in this browser. Use Chrome or Edge.';
        return;
    }
    state.recognition = new SpeechRecognition();
    state.recognition.continuous = true;
    state.recognition.interimResults = true;
    state.recognition.lang = 'en-US';
    state.recognition.onresult = (event) => {
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                const transcript = event.results[i][0].transcript.trim();
                if (transcript) sendLiveVoiceTurn(transcript);
            }
        }
    };
    state.recognition.onerror = (event) => {
        if (event.error === 'no-speech') return;
        if (event.error === 'not-allowed') {
            showToast('Microphone access denied.', 'error');
            stopRecording();
            micStatus.textContent = 'Microphone access denied';
        }
    };
    micBtn.addEventListener('click', toggleRecording);
}
async function startRealTimeVisualizer() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.mediaStream = stream;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        state.audioContext = new AudioContext();
        const source = state.audioContext.createMediaStreamSource(stream);
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 128;
        source.connect(state.analyser);
        const canvas = document.getElementById('audio-visualizer');
        const visualizerContainer = document.getElementById('visualizer-container');
        if (visualizerContainer) visualizerContainer.style.display = 'block';
        const canvasCtx = canvas.getContext('2d');
        const bufferLength = state.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        function drawWave() {
            if (!state.isRecording) return;
            state.visualizerAnimId = requestAnimationFrame(drawWave);
            state.analyser.getByteFrequencyData(dataArray);
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
            const barWidth = (canvas.width / bufferLength) * 1.5;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * canvas.height;
                const isHighStress = dataArray[i] > 180;
                canvasCtx.fillStyle = isHighStress ? '#ef4444' : '#4f6bf5';
                canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }
        drawWave();
    } catch (e) {
        console.warn('Real-time visualizer unavailable:', e);
    }
}
function stopRealTimeVisualizer() {
    if (state.visualizerAnimId) cancelAnimationFrame(state.visualizerAnimId);
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(t => t.stop());
        state.mediaStream = null;
    }
    if (state.audioContext) {
        state.audioContext.close();
        state.audioContext = null;
    }
    const visualizerContainer = document.getElementById('visualizer-container');
    if (visualizerContainer) visualizerContainer.style.display = 'none';
}
function toggleRecording() {
    if (!state.isRecording) {
        if (!state.recognition) {
            showToast('Speech recognition not supported in this browser.', 'error');
            return;
        }
        state.voiceMeetingId = null;
        document.getElementById('voice-live-feed').innerHTML = '';
        try {
            state.recognition.start();
            state.isRecording = true;
            document.getElementById('mic-btn').classList.add('recording');
            document.getElementById('mic-status').textContent = 'Listening live... Speak now. Click to stop and generate notes.';
            startRealTimeVisualizer();
            showToast('Live voice recording active', 'success');
        } catch (e) {
            console.error(e);
            showToast('Failed to start microphone', 'error');
        }
    } else {
        stopRecording();
    }
}
function stopRecording() {
    state.isRecording = false;
    if (state.recognition) state.recognition.stop();
    document.getElementById('mic-btn').classList.remove('recording');
    document.getElementById('mic-status').textContent = 'Recording stopped. Synthesizing notes...';
    stopRealTimeVisualizer();
    if (state.voiceMeetingId !== null) {
        endVoiceSession();
    }
}
async function sendLiveVoiceTurn(text) {
    const feed = document.getElementById('voice-live-feed');
    try {
        const response = await fetch('/api/analyze-sentence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                meeting_id: state.voiceMeetingId,
                privacy_mode: state.privacyMode
            })
        });
        const data = await response.json();
        if (data.meeting_id) state.voiceMeetingId = data.meeting_id;
        const turnDiv = document.createElement('div');
        turnDiv.className = `turn-card ${data.retained ? 'turn-kept' : 'turn-discarded'}`;
        turnDiv.innerHTML = `
            <div style="font-size:13px;"><strong>${escapeHtml(data.speaker || 'Speaker')}:</strong> ${escapeHtml(data.sentence)}</div>
            <div class="turn-badges" style="margin-top:6px;">
                <span class="badge ${data.retained ? 'badge-kept' : 'badge-discarded'}">${data.retained ? 'KEPT' : 'DISCARDED'}</span>
                <span style="font-size:11px;color:var(--text-secondary)">${data.classification.toUpperCase()} (${(data.confidence*100).toFixed(0)}%)</span>
            </div>
        `;
        feed.appendChild(turnDiv);
        turnDiv.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        console.error('Voice send error:', e);
    }
}
async function endVoiceSession() {
    try {
        const response = await fetch('/api/end-voice-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ meeting_id: state.voiceMeetingId })
        });
        const data = await response.json();
        state.currentResults = data;
        renderCorkboardStickyNotes(data.sticky_notes || []);
        showToast('Live session completed and Sticky Notes generated.', 'success');
        switchTab('corkboard');
    } catch (e) {
        console.error('End voice failed:', e);
    }
}
function renderCorkboardStickyNotes(notes) {
    const canvas = document.getElementById('corkboard-canvas');
    const emptyState = document.getElementById('empty-board-state');
    const summaryCard = document.getElementById('summary-card');
    const summaryBox = document.getElementById('summary-box');
    if (!notes || notes.length === 0) {
        if (emptyState) emptyState.style.display = 'block';
        return;
    }
    if (emptyState) emptyState.style.display = 'none';
    canvas.querySelectorAll('.sticky-note').forEach(n => n.remove());
    notes.forEach((note) => {
        const noteEl = document.createElement('div');
        noteEl.className = 'sticky-note';
        noteEl.style.backgroundColor = note.bg_color;
        noteEl.style.borderColor = note.border_color;
        noteEl.style.color = note.text_color;
        noteEl.style.transform = `rotate(${note.rotation || 0}deg)`;
        const bulletsHtml = (note.items || []).map(it => `<li>${formatNoteText(it)}</li>`).join('');
        noteEl.innerHTML = `
            <div class="pushpin ${note.pin_color || 'pin-red'}"></div>
            <div class="note-header">
                <span class="note-title">${escapeHtml(note.title)}</span>
                <span class="note-tag" style="color:${note.text_color};">${escapeHtml(note.tag || 'NOTE')}</span>
            </div>
            <ul class="note-bullets">
                ${bulletsHtml || '<li>No items listed.</li>'}
            </ul>
        `;
        canvas.appendChild(noteEl);
    });
    if (state.currentResults && state.currentResults.summary) {
        summaryCard.style.display = 'block';
        summaryBox.textContent = state.currentResults.summary;
    }
}
function formatNoteText(text) {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}
function renderWordWeightsHeatmap(data) {
    const container = document.getElementById('word-weights-container');
    if (!container) return;
    const weightsList = [];
    if (data.results) {
        data.results.forEach(turn => {
            if (turn.token_weights) {
                turn.token_weights.forEach(tw => weightsList.push(tw));
            }
        });
    }
    if (data.word_weights) {
        data.word_weights.forEach(ww => {
            weightsList.push({
                token: ww.word,
                weight: ww.weight,
                role: ww.stress_level || 'Acoustic Word',
                is_stressed: ww.is_stressed
            });
        });
    }
    if (weightsList.length === 0) {
        container.innerHTML = '<div class="empty-state">No word weights detected for this conversation.</div>';
        return;
    }
    const uniqueMap = {};
    weightsList.forEach(w => {
        const key = w.token.toLowerCase();
        if (!uniqueMap[key] || w.weight > uniqueMap[key].weight) {
            uniqueMap[key] = w;
        }
    });
    const sortedTokens = Object.values(uniqueMap).sort((a, b) => b.weight - a.weight);
    container.innerHTML = '';
    sortedTokens.slice(0, 45).forEach(tw => {
        const chip = document.createElement('div');
        const stressClass = tw.weight >= 0.85
            ? 'chip-high-stress'
            : (tw.weight >= 0.70 ? 'chip-technical' : 'chip-medium-stress');
        chip.className = `word-chip ${stressClass}`;
        chip.innerHTML = `
            <span>${escapeHtml(tw.token)}</span>
            <span class="chip-weight-tag">${(tw.weight * 100).toFixed(0)}% (${escapeHtml(tw.role || 'Token')})</span>
        `;
        container.appendChild(chip);
    });
}
function renderClassificationAudit(data) {
    const feed = document.getElementById('classification-feed');
    const stats = data.stats || {};
    const total = stats.total_segments_processed || data.total_sentences || 0;
    const kept = stats.professional_segments_retained || (data.kept_segments ? data.kept_segments.length : 0);
    const discarded = stats.private_segments_discarded || (data.discarded_segments ? data.discarded_segments.length : 0);
    const density = total > 0 ? ((kept / total) * 100).toFixed(1) : '0';
    document.getElementById('stat-total-turns').textContent = total;
    document.getElementById('stat-kept-turns').textContent = kept;
    document.getElementById('stat-discarded-turns').textContent = discarded;
    document.getElementById('stat-density').textContent = `${density}%`;
    document.getElementById('count-all').textContent = total;
    document.getElementById('count-kept').textContent = kept;
    document.getElementById('count-discarded').textContent = discarded;
    feed.innerHTML = '';
    const results = data.results || [];
    if (results.length === 0) {
        feed.innerHTML = '<div class="empty-state">No conversation data available.</div>';
        return;
    }
    results.forEach(turn => {
        const card = document.createElement('div');
        card.className = `turn-card ${turn.retained ? 'turn-kept' : 'turn-discarded'}`;
        card.dataset.retained = turn.retained ? 'kept' : 'discarded';
        let extractedBadges = '';
        if (turn.extracted) {
            const ext = turn.extracted;
            if (ext.tasks && ext.tasks.length) extractedBadges += `<span class="feature-tag">Task: ${escapeHtml(ext.tasks[0])}</span>`;
            if (ext.decisions && ext.decisions.length) extractedBadges += `<span class="feature-tag">Decision: ${escapeHtml(ext.decisions[0])}</span>`;
            if (ext.deadlines && ext.deadlines.length) extractedBadges += `<span class="feature-tag">Deadline: ${escapeHtml(ext.deadlines[0])}</span>`;
            if (ext.projects && ext.projects.length) extractedBadges += `<span class="feature-tag">${escapeHtml(ext.projects.join(', '))}</span>`;
        }
        const reasonTag = turn.discard_reason ? `<span style="color:var(--danger);font-size:11.5px;">[Reason: ${escapeHtml(turn.discard_reason)}]</span>` : '';
        card.innerHTML = `
            <div class="turn-header">
                <span class="turn-speaker">${escapeHtml(turn.speaker || 'Speaker')}</span>
                <div class="turn-badges">
                    <span class="badge ${turn.retained ? 'badge-kept' : 'badge-discarded'}">
                        ${turn.retained ? 'KEPT / IMPORTANT' : 'DISCARDED'}
                    </span>
                    <span class="badge ${getCategoryBadgeClass(turn.classification)}">${(turn.classification || 'UNCERTAIN').toUpperCase()}</span>
                </div>
            </div>
            <div class="turn-text">${escapeHtml(turn.sentence)}</div>
            <div class="turn-meta">
                <span>Confidence: <strong>${((turn.confidence || 0) * 100).toFixed(1)}%</strong></span>
                <span>Score: <strong>${((turn.professional_score || 0) * 100).toFixed(1)}%</strong></span>
                ${reasonTag}
            </div>
            ${extractedBadges ? `<div class="turn-features">${extractedBadges}</div>` : ''}
        `;
        feed.appendChild(card);
    });
    applyClassificationFilter(state.activeFilter);
}
function getCategoryBadgeClass(classification) {
    switch ((classification || '').toLowerCase()) {
        case 'professional': return 'badge-prof';
        case 'casual': return 'badge-casual';
        case 'personal': return 'badge-personal';
        default: return 'badge-casual';
    }
}
function initClassificationFilters() {
    document.querySelectorAll('.pill-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const filter = btn.dataset.filter;
            state.activeFilter = filter;
            applyClassificationFilter(filter);
        });
    });
}
function applyClassificationFilter(filter) {
    document.querySelectorAll('.turn-card').forEach(card => {
        if (filter === 'all') {
            card.style.display = 'block';
        } else if (filter === 'kept') {
            card.style.display = card.dataset.retained === 'kept' ? 'block' : 'none';
        } else if (filter === 'discarded') {
            card.style.display = card.dataset.retained === 'discarded' ? 'block' : 'none';
        }
    });
}
async function runEvaluationMetrics() {
    const reportContainer = document.getElementById('metrics-live-report');
    const btn = document.getElementById('run-metrics-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Running Evaluation Benchmark...';
    }
    reportContainer.innerHTML = '<div class="loading-spinner" style="margin:40px auto;"></div>';
    try {
        const response = await fetch('/api/evaluate-metrics', { method: 'POST' });
        const data = await response.json();
        if (data.error) {
            reportContainer.innerHTML = `<div class="empty-state">Evaluation error: ${escapeHtml(data.error)}</div>`;
            return;
        }
        const clf = data.classification_metrics || {};
        const r = data.rouge_scores || {};
        const b = data.bleu_scores || {};
        reportContainer.innerHTML = `
            <div class="card" style="margin-top:20px;">
                <div class="card-header">
                    <span class="card-title">1. Important-Content Classification Performance</span>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value stat-kept">${((clf.accuracy || 0)*100).toFixed(1)}%</div>
                        <div class="stat-label">Overall Accuracy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value stat-accent">${((clf.macro_f1 || 0)*100).toFixed(1)}%</div>
                        <div class="stat-label">Macro F1-Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value stat-warning">${clf.total_samples || 0}</div>
                        <div class="stat-label">Evaluated Sentences</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-title">2. Summary and Sticky Note Generation Quality (ROUGE and BLEU)</span>
                </div>
                <p class="card-description">
                    Measures n-gram overlap and longest common subsequence between generated sticky notes and gold standard meeting notes.
                </p>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" style="color:#d946ef;">${((r.rouge_1_f1 || 0)*100).toFixed(1)}%</div>
                        <div class="stat-label">ROUGE-1 (Unigram F1)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color:#06b6d4;">${((r.rouge_2_f1 || 0)*100).toFixed(1)}%</div>
                        <div class="stat-label">ROUGE-2 (Bigram F1)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color:#f59e0b;">${((r.rouge_l_f1 || 0)*100).toFixed(1)}%</div>
                        <div class="stat-label">ROUGE-L (LCS F1)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color:#3b82f6;">${((b.bleu_1 || 0)*100).toFixed(1)}%</div>
                        <div class="stat-label">BLEU-1 Precision</div>
                    </div>
                </div>
            </div>
        `;
        showToast('Evaluation benchmark completed', 'success');
    } catch (e) {
        console.error('Metrics failed:', e);
        reportContainer.innerHTML = '<div class="empty-state">Failed to load evaluation metrics.</div>';
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Run Metrics Benchmark';
        }
    }
}
async function loadMeetings() {
    const container = document.getElementById('meetings-list');
    const detail = document.getElementById('meeting-detail');
    container.style.display = 'block';
    detail.style.display = 'none';
    container.innerHTML = '<div class="loading-spinner" style="margin:40px auto;"></div>';
    try {
        const response = await fetch('/api/meetings');
        const data = await response.json();
        if (!Array.isArray(data) || data.length === 0) {
            container.innerHTML = '<div class="empty-state">No saved meeting boards found yet.</div>';
            return;
        }
        container.innerHTML = '';
        data.forEach(meeting => {
            const card = document.createElement('div');
            card.className = 'card';
            card.style.cursor = 'pointer';
            card.onclick = () => viewMeeting(meeting.id);
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="font-weight:700; font-size:15px; color:var(--text-primary);">${escapeHtml(meeting.session_name || 'Meeting')}</div>
                    <span class="badge badge-prof">ID: ${meeting.id}</span>
                </div>
                <div style="display:flex; gap:16px; font-size:13px; color:var(--text-secondary);">
                    <span>Date: ${escapeHtml(meeting.start_time || 'N/A')}</span>
                    <span style="color:var(--success)">Retained: ${meeting.professional_segments_retained || 0}</span>
                    <span style="color:var(--danger)">Discarded: ${meeting.discarded_segments_count || 0}</span>
                </div>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        container.innerHTML = '<div class="empty-state">Failed to load meeting history.</div>';
    }
}
async function viewMeeting(id) {
    try {
        const response = await fetch(`/api/meetings/${id}`);
        const data = await response.json();
        state.currentResults = data;
        renderCorkboardStickyNotes(data.sticky_notes || []);
        showToast(`Loaded Meeting ${id} Sticky Notes`, 'success');
        switchTab('corkboard');
    } catch (e) {
        showToast('Failed to load meeting board', 'error');
    }
}
async function loadPrivacyMode() {
    try {
        const response = await fetch('/api/privacy-mode');
        const data = await response.json();
        state.privacyMode = data.mode || 'BALANCED';
        document.getElementById('header-mode').textContent = state.privacyMode;
    } catch (e) {
        console.error(e);
    }
}
async function setPrivacyMode(mode) {
    const threshold = mode === 'CUSTOM' ? parseFloat(document.getElementById('custom-threshold').value) : undefined;
    try {
        const response = await fetch('/api/privacy-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, custom_threshold: threshold })
        });
        const data = await response.json();
        state.privacyMode = data.mode || mode;
        document.getElementById('header-mode').textContent = state.privacyMode;
        document.querySelectorAll('.mode-card').forEach(opt => {
            opt.classList.toggle('selected', opt.dataset.mode === state.privacyMode);
        });
        showToast(`Privacy mode set to ${state.privacyMode}`, 'success');
    } catch (e) {
        showToast('Failed to update privacy mode', 'error');
    }
}
function exportSummary() {
    if (!state.currentResults || !state.currentResults.summary) {
        showToast('No meeting summary to export.', 'error');
        return;
    }
    const blob = new Blob([state.currentResults.summary], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meetlytic-summary-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    showToast('Report exported as Markdown', 'success');
}
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}
document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchTab(e.currentTarget.dataset.tab);
        });
    });
    initSubtabs();
    initAudioDropZone();
    initSpeechRecognition();
    initClassificationFilters();
    loadPrivacyMode();
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeTranscript);
    document.querySelectorAll('.mode-card').forEach(opt => {
        opt.addEventListener('click', (e) => {
            if (e.target.tagName === 'INPUT') return;
            setPrivacyMode(opt.dataset.mode);
        });
    });
});
