import type { Meta, StoryObj } from "@storybook/react-vite";
import { ThemeToggle } from "./ThemeToggle";

const meta = { component: ThemeToggle, args: { onFlip: () => {} } } satisfies Meta<typeof ThemeToggle>;
export default meta;
type Story = StoryObj<typeof meta>;

export const InLight: Story = { args: { shown: "light" } };
export const InDark: Story = { args: { shown: "dark" } };
