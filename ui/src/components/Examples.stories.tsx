import type { Meta, StoryObj } from "@storybook/react-vite";
import { Examples } from "./Examples";

const meta = { component: Examples, args: { onPick: () => {} } } satisfies Meta<typeof Examples>;
export default meta;
type Story = StoryObj<typeof meta>;

export const SixSearches: Story = {
  args: {
    examples: [
      "How much revenue did IBM generate from Mobile in the third quarter of 2018?",
      "For how many years must Supplier keep records of financial information?",
      "What is the barista annual salary based on assumptions?",
      "Which form lists acceptable documents in List A, List B and List C?",
      "ILO Convention C189 ratification EU countries 2024",
      "What date had the most ridership?",
    ],
  },
};
