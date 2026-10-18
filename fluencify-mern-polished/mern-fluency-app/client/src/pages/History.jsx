import { useEffect, useState } from "react";
import api from "../api/client.js";

export default function History() {
  const [attempts, setAttempts] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .get("/history")
      .then(({ data }) => setAttempts(data))
      .catch((err) => setError(err.response?.data?.error || "Failed to load history"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container">
      <div className="card">
        <h2>My attempts</h2>
        {error && <div className="error">{error}</div>}
        {loading ? (
          <p className="muted">Loading...</p>
        ) : attempts.length === 0 ? (
          <p className="muted">No attempts yet — go record or upload one.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>File</th>
                <th>Score</th>
                <th>Dominant issue</th>
              </tr>
            </thead>
            <tbody>
              {attempts.map((a) => (
                <tr key={a._id}>
                  <td>{new Date(a.createdAt).toLocaleString()}</td>
                  <td>{a.originalFileName || "recording.webm"}</td>
                  <td>{a.fluencyScore}</td>
                  <td>{a.summary?.dominantIssue || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
