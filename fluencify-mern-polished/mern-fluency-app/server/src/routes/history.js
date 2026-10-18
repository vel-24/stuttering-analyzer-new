import { Router } from "express";
import { requireAuth } from "../middleware/auth.js";
import Attempt from "../models/Attempt.js";

const router = Router();

// GET /api/history  -> list the current user's past attempts, newest first
router.get("/", requireAuth, async (req, res) => {
  const attempts = await Attempt.find({ user: req.userId })
    .sort({ createdAt: -1 })
    .limit(200);
  res.json(attempts);
});

// GET /api/history/:id -> a single attempt's full detail (segments, breakdown)
router.get("/:id", requireAuth, async (req, res) => {
  const attempt = await Attempt.findOne({ _id: req.params.id, user: req.userId });
  if (!attempt) return res.status(404).json({ error: "Attempt not found" });
  res.json(attempt);
});

// DELETE /api/history/:id
router.delete("/:id", requireAuth, async (req, res) => {
  const result = await Attempt.deleteOne({ _id: req.params.id, user: req.userId });
  if (result.deletedCount === 0) return res.status(404).json({ error: "Attempt not found" });
  res.status(204).end();
});

export default router;
