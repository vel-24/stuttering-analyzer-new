import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function ScoreBreakdownChart({ breakdown }) {
  if (!breakdown) return null;
  const data = Object.entries(breakdown).map(([key, value]) => ({
    name: key.replace(/([A-Z])/g, " $1").trim(),
    value,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout="vertical" margin={{ left: 40 }}>
        <XAxis type="number" stroke="#9aa1ad" />
        <YAxis type="category" dataKey="name" stroke="#9aa1ad" width={140} />
        <Tooltip contentStyle={{ background: "#171a21", border: "1px solid #262a33" }} />
        <Bar dataKey="value" fill="#4f7cff" radius={[0, 6, 6, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
