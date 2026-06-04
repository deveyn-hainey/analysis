import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SoccerVision Analytics",
  description: "Computer vision–powered soccer match analysis for coaching staff",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0d1117] text-[#e6edf3] antialiased">
        {children}
      </body>
    </html>
  );
}
