# Zap Website

Astro site, derived from the visual mockups under `design/`.

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # outputs dist/
```

Structure:

- `src/pages/index.astro` — Landing
- `src/pages/docs/[...slug].astro` — docs dynamic route
- `src/content/docs/*.mdx` — docs content (Content Collections)
- `src/components/` — Nav / Footer / Banner etc.
- `src/styles/` — design tokens and global styles
