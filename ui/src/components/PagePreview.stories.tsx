import type { Meta, StoryObj } from "@storybook/react-vite";
import { RESPONSE } from "../test/fixtures";
import { PagePreview } from "./PagePreview";

const meta = { component: PagePreview, args: { terms: RESPONSE.terms, onClose: () => {} } } satisfies Meta<
  typeof PagePreview
>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Slide: Story = { args: { group: RESPONSE.results.groups[0], page: 4 } };
export const Spreadsheet: Story = { args: { group: RESPONSE.results.groups[1], page: null } };
