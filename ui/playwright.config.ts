import { defineConfig, devices } from "@playwright/test";

// The built page (bun run build, or the build in UI_DIST) served by `vite preview`, with the API mocked per test:
// light and dark, desktop and phone. `docsearch serve` with the real API is checked by scripts/verify.sh.
const dist = process.env.UI_DIST ?? "dist";
const desktop = { viewport: { width: 1280, height: 800 } };
const phone = { ...devices["iPhone 13"], defaultBrowserType: "chromium" as const };

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  reporter: [["list"]],
  outputDir: process.env.UI_E2E_OUTPUT ?? "test-results",
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: {
    command: `bunx vite preview --host 127.0.0.1 --port 4173 --strictPort --outDir ${dist}`,
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
  },
  projects: [
    { name: "light-desktop", use: { ...desktop, colorScheme: "light" } },
    { name: "dark-desktop", use: { ...desktop, colorScheme: "dark" } },
    { name: "light-phone", use: { ...phone, colorScheme: "light" } },
    { name: "dark-phone", use: { ...phone, colorScheme: "dark" } },
  ],
});
