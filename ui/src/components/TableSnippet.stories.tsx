import type { Meta, StoryObj } from "@storybook/react-vite";
import { tableRows } from "../lib/table";
import { TableSnippet } from "./TableSnippet";

const meta = { component: TableSnippet } satisfies Meta<typeof TableSnippet>;
export default meta;
type Story = StoryObj<typeof meta>;
const table = (s: string) => tableRows(s) ?? { before: false, after: false, rows: [] };

export const Whole: Story = {
  args: {
    table: table("Barista Annual Salary | £50,000 | / Cost inflation | 0.05 | *"),
    terms: ["barista", "salary"],
  },
};
export const CutAtBothEnds: Story = {
  args: { table: table("…Counter | | 1 / Barista Salaries | | £50,000 | 0.05 | £4,167 |…"), terms: ["barista"] },
};
