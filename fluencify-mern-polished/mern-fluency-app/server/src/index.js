import "dotenv/config";
import express from "express";
import cors from "cors";
import morgan from "morgan";
import rateLimit from "express-rate-limit";

import { connectDB } from "./config/db.js";
import authRoutes from "./routes/auth.js";
import analyzeRoutes from "./routes/analyze.js";
import historyRoutes from "./routes/history.js";

const app = express();
app.set("trust proxy", 1);
app.use(cors({ origin: process.env.CLIENT_ORIGIN || "*" }));
app.use(express.json());
app.use(morgan("dev"));

// Basic brute-force protection on auth endpoints
const authLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 50 });
app.use("/api/auth", authLimiter, authRoutes);

// Analysis is the expensive call; cap it per IP so a public deployment can't be flooded.
const analyzeLimiter = rateLimit({ windowMs: 60 * 60 * 1000, max: 100 });
app.use("/api/analyze", analyzeLimiter, analyzeRoutes);
app.use("/api/history", historyRoutes);

app.get("/api/health", (_req, res) => res.json({ status: "ok" }));

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error" });
});

const PORT = process.env.PORT || 5000;

connectDB()
  .then(() => {
    app.listen(PORT, () => console.log(`Server listening on http://localhost:${PORT}`));
  })
  .catch((err) => {
    console.error("Failed to connect to MongoDB:", err.message);
    process.exit(1);
  });
