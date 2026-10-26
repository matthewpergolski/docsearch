import type { Meta, StoryObj } from "@storybook/react-vite";
import { SlideProgress } from "./SlideProgress";

const meta = { component: SlideProgress } satisfies Meta<typeof SlideProgress>;
export default meta;
type Story = StoryObj<typeof meta>;

const pages = Array.from({ length: 10 }, (_, i) => ({
  unit: `deck_page_${i + 1}#p1`,
  file: `deck_page_${i + 1}`,
  page: 1,
}));
export const ThreeOfTen: Story = { args: { reading: { pages, done: pages.slice(0, 3).map((p) => p.unit) } } };
