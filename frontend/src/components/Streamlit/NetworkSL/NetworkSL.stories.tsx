// stories/NetworkSLApp.stories.js
import React from "react";
import { Meta, Story } from "@storybook/react";
import data from "../../Network/examples";
import { NetworkSL, NetworkSLInterface } from "./NetworkSL";

export default {
  title: "Components/NetworkSL",
  component: NetworkSL,
  argTypes: {
    network: { control: "object" },
  },
} as Meta;

const Template: Story<NetworkSLInterface> = (args) => {
  return <NetworkSL {...args} />;
};

export const Default = Template.bind({});

Default.args = {
  network: data[0],
  maxMoves: 8,
  showAllEdges: false,
};
