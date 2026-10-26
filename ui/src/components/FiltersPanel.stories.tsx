import type { Meta, StoryObj } from "@storybook/react-vite";
import { NO_FILTERS } from "../types";
import { FiltersPanel } from "./FiltersPanel";

const FORMATS = ["csv", "docx", "image", "pdf", "pptx", "slide_image", "xlsx"];
const meta = { component: FiltersPanel, args: { formats: FORMATS, onChange: () => {} } } satisfies Meta<
  typeof FiltersPanel
>;
export default meta;
type Story = StoryObj<typeof meta>;

export const NoneSet: Story = { args: { value: NO_FILTERS } };
export const AllSet: Story = {
  args: {
    value: { formats: ["pdf", "xlsx"], after: "2024-01-01", before: "2024-12-31", folder: "finance", undated: true },
  },
};
