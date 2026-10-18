import { Router } from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import { requireAuth } from "../middleware/auth.js";
import Attempt from "../models/Attempt.js";

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB, matches inference service
});

const VALID_MODELS = [
  "whisper_ft_head",
  "fusion_ft",
  "whisper_ft_only",
  "fusion",
  "whisper_only",
  "acoustic_only",
];

// POST /api/analyze  (multipart/form-data: "audio" file, optional "model", "ftSeed")
// Forwards the uploaded file to the Python FastAPI inference service -- along with
// this user's recent attempts, so the service's personalized-feedback logic
// (ported from tracker.py) has something to compare against -- then saves the
// result against the authenticated user's history.
router.post("/", requireAuth, upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file uploaded (field name must be 'audio')" });
  }

  const model = VALID_MODELS.includes(req.body.model) ? req.body.model : "whisper_ft_head";
  const ftSeed = Number.isInteger(Number(req.body.ftSeed)) ? Number(req.body.ftSeed) : 0;

  try {
    // Last 5 attempts for this user, oldest first, shaped the way fluency.py's
    // WEIGHTS-keyed `components` + score are expected (tracker.feedback()'s contract).
    const recent = await Attempt.find({ user: req.userId })
      .sort({ createdAt: -1 })
      .limit(5)
      .lean();
    const prior = recent
      .reverse()
      .map((a) => ({ score: a.fluencyScore, components: a.scoreBreakdown }));

    const form = new FormData();
    form.append("audio", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    form.append("model", model);
    form.append("ft_seed", String(ftSeed));
    form.append("prior", JSON.stringify(prior));

    const inferenceUrl = `${process.env.INFERENCE_SERVICE_URL}/analyze`;
    const { data: result } = await axios.post(inferenceUrl, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 120_000,
    });

    const attempt = await Attempt.create({
      user: req.userId,
      originalFileName: req.file.originalname,
      model: result.modelUsed,
      fluencyScore: result.fluencyScore,
      scoreBreakdown: result.scoreBreakdown,
      segments: result.segments,
      summary: result.summary,
      feedback: result.feedback,
      stub: result.stub,
    });

    res.status(201).json(attempt);
  } catch (err) {
    if (err.response) {
      // Inference service responded with an error (bad file, model failure, etc.)
      return res.status(502).json({
        error: "Inference service error",
        detail: err.response.data,
      });
    }
    res.status(500).json({ error: "Analysis failed", detail: err.message });
  }
});

export default router;
