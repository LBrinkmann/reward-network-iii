// stories/NetworkSLApp.stories.js
import React from "react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { Meta, Story } from "@storybook/react";
import data from "../../Network/examples";
import NetworkSLApp from "./NetworkSL";
// to json
const network_str = JSON.stringify(data[0]);

export default {
  title: "Components/NetworkSLApp",
  component: NetworkSLApp,
  argTypes: {
    network: { control: "object" },
  },
} as Meta;

const Template: Story = (args) => {
  const searchParams = new URLSearchParams();
  searchParams.set("network", JSON.stringify(args.network));
  searchParams.set("max_moves", args.max_moves);
  searchParams.set("showAllEdges", args.showAllEdges);
  return (
    <MemoryRouter initialEntries={[`/streamlit?${searchParams.toString()}`]}>
      <Routes>
        <Route path="/streamlit" element={<NetworkSLApp {...args} />} />
      </Routes>
    </MemoryRouter>
  );
};

export const Default = Template.bind({});

Default.args = {
  network: JSON.parse(network_str),
  max_moves: 8,
  showAllEdges: false,
};
