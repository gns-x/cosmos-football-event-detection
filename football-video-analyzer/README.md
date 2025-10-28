# Football Video Analysis - NVIDIA Cosmos Interface

A Next.js application for AI-powered football video analysis using the Cosmos-Reason1-7B model, featuring a two-column layout matching NVIDIA's design system.

## Features

- **Two-Column Layout**: Video player on the left, controls on the right
- **NVIDIA Design System**: Matches build.nvidia.com styling with proper colors and components
- **Video Player**: 30-second timeline with play/pause controls
- **Interactive Prompts**: User and System prompt cards with character counters
- **Output Display**: Preview and JSON tabs for model responses
- **Responsive Design**: Mobile-friendly layout that stacks on smaller screens

## Design Specifications

### Colors
- **NVIDIA Green**: #76B900 (primary accent)
- **Background**: #0D1117 (dark theme)
- **Card Background**: #161B22
- **Card Border**: #30363D
- **Text Primary**: #F0F6FC
- **Text Secondary**: #8B949E

### Layout
- **Left Column**: 60-65% width on desktop, full width on mobile
- **Right Column**: Fixed 384px width (w-96) on desktop, full width on mobile
- **Card Styling**: Rounded corners, subtle shadows, consistent padding
- **Spacing**: 24px vertical rhythm between cards

## Getting Started

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Run the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## Project Structure

```
src/
├── app/
│   ├── globals.css          # NVIDIA-themed global styles
│   ├── layout.tsx          # Root layout with metadata
│   └── page.tsx            # Main page component
└── components/
    ├── VideoPlayer.tsx      # Video player with controls
    ├── PromptCard.tsx      # Reusable prompt input card
    └── OutputSection.tsx   # Output display with tabs
```

## Components

### VideoPlayer
- Custom video player with NVIDIA styling
- 30-second timeline with progress indicator
- Play/pause controls with hover effects
- Time display in MM:SS format

### PromptCard
- Textarea input with character counter
- Dynamic styling based on character limit
- Consistent card design with other components

### OutputSection
- Tabbed interface (Preview/JSON)
- Structured output display
- JSON formatting with syntax highlighting

## Customization

The application uses CSS custom properties for easy theming:

```css
:root {
  --nvidia-green: #76B900;
  --nvidia-dark-bg: #0D1117;
  --nvidia-card-bg: #161B22;
  --nvidia-card-border: #30363D;
  --nvidia-text-primary: #F0F6FC;
  --nvidia-text-secondary: #8B949E;
  --nvidia-accent: #58A6FF;
}
```

## Technologies Used

- **Next.js 16.0.1**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **React Hooks**: State management and side effects

## Browser Support

- Chrome/Edge 88+
- Firefox 85+
- Safari 14+
- Mobile browsers with modern JavaScript support

## License

This project is part of the NVIDIA Cosmos football video analysis pipeline.