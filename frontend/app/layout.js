import './globals.css';

export const metadata = {
  title: 'DeepScan - AI Deepfake Detector',
  description: 'Advanced AI-powered deepfake detection using CNN analysis with heatmap and Fourier visualization',
  keywords: ['deepfake', 'detection', 'AI', 'CNN', 'image analysis'],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="bg-ds-primary text-ds-text antialiased">
        {children}
      </body>
    </html>
  );
}
