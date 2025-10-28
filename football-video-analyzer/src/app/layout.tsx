import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Football Video Analysis - NVIDIA Cosmos",
  description: "AI-powered football event detection using Cosmos-Reason1-7B model",
  keywords: ["football", "video analysis", "AI", "NVIDIA", "Cosmos", "machine learning"],
  authors: [{ name: "NVIDIA" }],
  viewport: "width=device-width, initial-scale=1",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}