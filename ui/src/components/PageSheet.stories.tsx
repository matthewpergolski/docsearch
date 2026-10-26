import type { Meta, StoryObj } from "@storybook/react-vite";
import { RESPONSE } from "../test/fixtures";
import { PageSheet } from "./PageSheet";

const meta = { component: PageSheet, decorators: [(S) => <div style={{ width: 144 }}>{S()}</div>] } satisfies Meta<
  typeof PageSheet
>;
export default meta;
type Story = StoryObj<typeof meta>;

export const SlideWithNoPicture: Story = { args: { group: RESPONSE.results.groups[0] } };
export const Spreadsheet: Story = { args: { group: RESPONSE.results.groups[1] } };
