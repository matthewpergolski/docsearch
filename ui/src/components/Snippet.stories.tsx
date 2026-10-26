import type { Meta, StoryObj } from "@storybook/react-vite";
import { Snippet } from "./Snippet";

const meta = { component: Snippet } satisfies Meta<typeof Snippet>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Highlighted: Story = {
  args: {
    text: "Freight cost per region rose 4% in Q3 2024; region totals follow.",
    terms: ["freight", "region", "2024"],
  },
};
export const NoTerms: Story = { args: { text: "A snippet with nothing to highlight.", terms: [] } };
