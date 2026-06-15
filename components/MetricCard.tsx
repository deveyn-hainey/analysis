import type { LucideIcon } from "lucide-react";

interface MetricCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  sub?: string;
}

export default function MetricCard({ icon: Icon, label, value, sub }: MetricCardProps) {
  return (
    <div className="card p-5">
      <div className="w-9 h-9 rounded-lg bg-green-400/10 flex items-center justify-center mb-3">
        <Icon className="text-green-400" style={{ width: 18, height: 18 }} />
      </div>
      <div className="text-2xl font-bold text-[#f0fdf4] mb-0.5">{value}</div>
      <div className="text-sm text-[#6b9e6b]">{label}</div>
      {sub && <div className="text-xs text-[#6b9e6b] mt-1 opacity-70">{sub}</div>}
    </div>
  );
}
