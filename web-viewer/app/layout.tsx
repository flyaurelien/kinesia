import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Kinesia",
  description: "Freezing-of-gait detection and 3D gait analysis viewer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* No external fonts/CDNs: the app must run fully offline. The CSS
            font stack falls back to the system UI font (Inter, ui-sans-serif,
            system-ui, …) without any network request. */}
        <link rel="icon" href="/favicon.png" type="image/png" />
      </head>
      <body>{children}</body>
    </html>
  );
}
