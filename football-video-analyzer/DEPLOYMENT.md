# Football Video Analysis - Deployment Configuration

## Environment Variables

Create a `.env.local` file in the project root:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_MODEL_NAME=Cosmos-Reason1-7B

# Optional: External API endpoints
COSMOS_API_URL=your-cosmos-api-endpoint
COSMOS_API_KEY=your-api-key
```

## Deployment Options

### 1. Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### 2. Docker
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

### 3. Static Export
```bash
npm run export
# Deploy the 'out' folder to any static hosting
```

## Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Performance Optimization

- Images are automatically optimized by Next.js
- CSS is automatically minified
- JavaScript bundles are code-split
- Static assets are cached with proper headers

## Security Considerations

- API routes are protected by default
- Environment variables are server-side only
- CSRF protection is enabled
- Content Security Policy headers are recommended
