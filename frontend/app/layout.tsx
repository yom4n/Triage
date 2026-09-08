import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FDE Incident Triage",
  description: "Operational frontend for the FDE triage backend"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
