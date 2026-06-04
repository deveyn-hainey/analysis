import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  sub?: string;
  accent?: "emerald" | "blue" | "red" | "yellow";
}

const ACCENT_CLASSES = {
  emerald: { bg: "bg-emerald-400/10", text: "text-emerald-400" },
  blue: { bg: "bg-blue-400/10", text: "text-blue-400" },
  red: { bg: "bg-red-400/10", text: "text-red-400" },
  yellow: { bg: "bg-yellow-400/10", text: "text-yellow-400" },
};

export default function MetricCard({ icon: Icon, label, value, sub, accent = "emerald" }: MetricCardProps) {
  const cls = ACCENT_CLASSES[accent];
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between mb-3">
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${cls.bg}`}>
          <Icon className={`w-4.5 h-4.5 ${cls.text}`} style={{ width: 18, height: 18 }} />
        </div>
      </div>
      <div className="text-2xl font-bold text-[#e6edf3] mb-0.5">{value}</div>
      <div className="text-sm text-[#8b949e]">{label}</div>
      {sub && <div className="text-xs text-[#8b949e] mt-1 opacity-70">{sub}</div>}
    </div>
  );
}
