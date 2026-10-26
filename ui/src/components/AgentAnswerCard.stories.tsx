import type { Meta, StoryObj } from "@storybook/react-vite";
import { AgentAnswerCard } from "./AgentAnswerCard";

const meta = { component: AgentAnswerCard } satisfies Meta<typeof AgentAnswerCard>;
export default meta;
type Story = StoryObj<typeof meta>;

export const WithValue: Story = {
  args: { answer: { value: "$4.2M", file: "reports/supplier-deck-q3.pptx", page: 4 } },
};
export const PageOnly: Story = { args: { answer: { value: "", file: "finance/costs-2024.xlsx", page: null } } };
