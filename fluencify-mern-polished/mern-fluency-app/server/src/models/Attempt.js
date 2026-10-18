import mongoose from "mongoose";

// Mirrors the shape returned by the FastAPI /analyze endpoint
// (see inference-service/pipeline_adapter.py for the current stub shape;
// this schema is intentionally loose via Mixed so it doesn't need to
// change every time the real pipeline's output evolves).
const attemptSchema = new mongoose.Schema(
  {
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    originalFileName: { type: String },
    model: { type: String },
    fluencyScore: { type: Number, required: true },
    scoreBreakdown: { type: mongoose.Schema.Types.Mixed },
    segments: { type: mongoose.Schema.Types.Mixed },
    summary: { type: mongoose.Schema.Types.Mixed },
    feedback: { type: [String], default: [] },
    stub: { type: Boolean, default: false }, // true while inference-service's real model artifacts aren't in place yet
  },
  { timestamps: true }
);

export default mongoose.model("Attempt", attemptSchema);
