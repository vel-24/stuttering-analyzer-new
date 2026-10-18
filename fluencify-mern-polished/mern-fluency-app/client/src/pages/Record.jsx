import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/client.js";
import ScoreBreakdownChart from "../components/ScoreBreakdownChart.jsx";

// Inference configuration stays internal so users can focus on their recording.
const MODEL = "whisper_ft_head";
const FT_SEED = 0;

export default function Record() {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const startedAtRef = useRef(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  function clearPreviousAudio() {
    setResult(null);
    setError("");
    setRecordingSeconds(0);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
  }

  async function startRecording() {
    clearPreviousAudio();
    setAudioBlob(null);
    setUploadedFile(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Recording isn't available in this browser. Please upload an audio file instead.");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      startedAtRef.current = Date.now();
      recorder.ondataavailable = (event) => {
        if (event.data?.size) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((track) => track.stop());
        if (timerRef.current) clearInterval(timerRef.current);
        setRecordingSeconds(Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000)));
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
      timerRef.current = setInterval(() => {
        setRecordingSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
      }, 500);
    } catch {
      setError("Microphone access was denied or is unavailable. You can upload an audio file instead.");
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current?.state !== "inactive") mediaRecorderRef.current?.stop();
    setRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setUploadedFile(file);
    setAudioBlob(null);
    setAudioUrl(URL.createObjectURL(file));
    setResult(null);
    setError("");
    setRecordingSeconds(0);
  }

  function removeAudio() {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioBlob(null);
    setUploadedFile(null);
    setAudioUrl(null);
    setResult(null);
    setError("");
    setRecordingSeconds(0);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleAnalyze() {
    const file = uploadedFile || audioBlob;
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      const filename = uploadedFile ? uploadedFile.name : "recording.webm";
      form.append("audio", file, filename);
      form.append("model", MODEL);
      form.append("ftSeed", String(FT_SEED));
      const { data } = await api.post("/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.error || "We couldn't analyze this recording. Please try another audio file.");
    } finally {
      setLoading(false);
    }
  }

  const hasAudio = Boolean(audioBlob || uploadedFile);
  const audioName = uploadedFile?.name || (audioBlob ? "Your voice recording" : "");

  return (
    <main className="record-page">
      <section className="record-hero">
        <div className="hero-copy">
          <span className="eyebrow"><span className="eyebrow-dot" /> YOUR SPEECH, UNDERSTOOD</span>
          <h1>Make space for<br /><span>your voice.</span></h1>
          <p>Record a short sample or upload audio to explore your speech fluency and get supportive, personalized feedback.</p>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="sound-orbit orbit-one" />
          <div className="sound-orbit orbit-two" />
          <div className="sound-core"><span>〰</span></div>
          <div className="sound-bars"><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
        </div>
      </section>

      <section className="workspace-card">
        <div className="section-heading">
          <div className="section-icon">♫</div>
          <div>
            <h2>Analyze a recording</h2>
            <p>Choose how you’d like to add your voice sample.</p>
          </div>
        </div>

        {error && <div className="notice notice-error" role="alert"><span>!</span><div>{error}</div></div>}

        <div className="capture-options">
          <button type="button" className={`capture-option ${recording ? "capture-recording" : ""}`} onClick={recording ? stopRecording : startRecording} disabled={loading}>
            <span className="capture-icon">{recording ? "■" : "●"}</span>
            <span className="capture-option-copy">
              <strong>{recording ? "Stop recording" : "Record your voice"}</strong>
              <small>{recording ? `Recording · ${recordingSeconds}s` : "Use your microphone"}</small>
            </span>
            {recording && <span className="recording-pulse" />}
          </button>
          <button type="button" className="capture-option" onClick={() => fileInputRef.current?.click()} disabled={loading || recording}>
            <span className="capture-icon upload-icon">↑</span>
            <span className="capture-option-copy">
              <strong>Upload an audio file</strong>
              <small>Choose an audio file from your device</small>
            </span>
          </button>
          <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileChange} hidden />
        </div>

        {audioUrl && (
          <div className="audio-preview">
            <div className="audio-file-icon">♫</div>
            <div className="audio-preview-main">
              <div className="audio-file-name" title={audioName}>{audioName}</div>
              <div className="audio-file-meta">Ready to analyze{recordingSeconds ? ` · ${recordingSeconds} sec` : ""}</div>
              <audio controls src={audioUrl} />
            </div>
            <button className="remove-audio" type="button" onClick={removeAudio} disabled={loading} aria-label="Remove audio" title="Remove audio">×</button>
          </div>
        )}

        <div className="analysis-actions">
          <div className="privacy-note"><span>▣</span> Your recording is analyzed to generate your results.</div>
          <button className="analyze-button" type="button" onClick={handleAnalyze} disabled={!hasAudio || loading || recording}>
            {loading ? <><span className="button-spinner" /> Analyzing your speech…</> : <>Analyze recording <span aria-hidden="true">→</span></>}
          </button>
        </div>
      </section>

      {result && (
        <section className="results-card">
          {result.stub && (
            <div className="notice notice-error" role="alert">
              <span>!</span><div><strong>Demo result</strong><br />The inference service returned demo data instead of a real model prediction. Check that the model artifacts are correctly configured before relying on this result.</div>
            </div>
          )}
          <div className="results-heading">
            <div>
              <span className="eyebrow">YOUR ANALYSIS</span>
              <h2>Your fluency snapshot</h2>
              <p>A starting point for reflecting on your speech. This is not a clinical diagnosis.</p>
            </div>
            <div className="score-circle"><strong>{result.fluencyScore ?? "—"}</strong><span>fluency score</span></div>
          </div>
          <div className="result-stats">
            <div><span>Segments reviewed</span><strong>{result.summary?.totalSegments ?? "—"}</strong></div>
            <div><span>Audio duration</span><strong>{result.summary?.durationSec != null ? `${result.summary.durationSec}s` : "—"}</strong></div>
            <div><span>Most frequent pattern</span><strong>{result.summary?.dominantIssue || "None detected"}</strong></div>
          </div>
          <div className="chart-wrap"><ScoreBreakdownChart breakdown={result.scoreBreakdown} /></div>
          {result.feedback?.length > 0 && (
            <div className="feedback-section">
              <h3>Helpful next steps</h3>
              <p className="result-subtitle">Small, supportive ideas you can try at your own pace.</p>
              <div className="feedback-list">
                {result.feedback.map((message, index) => <div className="feedback-item" key={index}><span>{String(index + 1).padStart(2, "0")}</span><p>{message}</p></div>)}
              </div>
            </div>
          )}
          <button className="secondary-button" type="button" onClick={() => navigate("/history")}>View analysis history <span>→</span></button>
        </section>
      )}
      <p className="disclaimer">Fluencify is designed for self-reflection and awareness. Results may be imperfect and should not replace advice from a qualified speech-language professional.</p>
    </main>
  );
}
