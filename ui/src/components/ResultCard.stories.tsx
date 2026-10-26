import type { Meta, StoryObj } from "@storybook/react-vite";
import { RESPONSE } from "../test/fixtures";
import { ResultCard } from "./ResultCard";

const meta = { component: ResultCard } satisfies Meta<typeof ResultCard>;
export default meta;
type Story = StoryObj<typeof meta>;

export const DeckWithManyPages: Story = { args: { group: RESPONSE.results.groups[0], terms: RESPONSE.terms } };
export const SpreadsheetWithoutPages: Story = { args: { group: RESPONSE.results.groups[1], terms: RESPONSE.terms } };
