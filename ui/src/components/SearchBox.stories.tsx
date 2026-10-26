import type { Meta, StoryObj } from "@storybook/react-vite";
import { SearchBox } from "./SearchBox";

const none = () => {};
const meta = {
  component: SearchBox,
  args: {
    query: "",
    filters: 0,
    filtersOpen: false,
    onType: none,
    onSubmit: none,
    onClear: none,
    onToggleFilters: none,
  },
} satisfies Meta<typeof SearchBox>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {};
export const WithQuery: Story = { args: { query: "Q3 freight cost by region" } };
export const Searching: Story = { args: { query: "Q3 freight cost by region", busy: true } };
export const TwoFilters: Story = { args: { query: "Q3 freight cost by region", filters: 2, filtersOpen: true } };
