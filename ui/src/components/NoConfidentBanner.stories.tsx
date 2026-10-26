import type { Meta, StoryObj } from "@storybook/react-vite";
import { NoConfidentBanner } from "./NoConfidentBanner";

const meta = { component: NoConfidentBanner } satisfies Meta<typeof NoConfidentBanner>;
export default meta;

export const Default: StoryObj<typeof meta> = {};
