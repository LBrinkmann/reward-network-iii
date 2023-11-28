import React, { useEffect } from "react";

import { ComponentMeta, ComponentStory } from "@storybook/react";

import Repeat, { IRepeat } from "./Repeat";

import data from "../../Network/examples";
import useNetworkContext, {
  NetworkContextProvider,
} from "../../../contexts/NetworkContext";

export default {
  title: "Trials/Repeat",
  component: Repeat,
  decorators: [
    (ComponentStory) => {
      return (
        <NetworkContextProvider saveToLocalStorage={false}>
          <ComponentStory />
        </NetworkContextProvider>
      );
    },
  ],
} as ComponentMeta<typeof Repeat>;

interface TemplateArgs extends IRepeat {
  wrongRepeatPunishment: number;
  correctRepeatReward: number;
}

const Template: ComponentStory<typeof Repeat> = function (args: TemplateArgs) {
  const { networkState, networkDispatcher } = useNetworkContext();
  const {
    solution,
    wrongRepeatPunishment,
    correctRepeatReward,
    playerTotalPoints,
  } = args;

  useEffect(() => {
    if (!networkState.network) {
      networkDispatcher({
        type: "setNetwork",
        payload: {
          network: {
            edges: data[0].edges,
            nodes: data[0].nodes,
          },
          isPractice: false,
          solution: {
            moves: solution,
          },
          correctRepeatReward: correctRepeatReward,
          wrongRepeatPunishment: wrongRepeatPunishment,
          trailType: "repeat",
        },
      });
    }
  }, []);

  return (
    <>
      {networkState.network && (
        <Repeat
          solution={solution}
          playerTotalPoints={playerTotalPoints}
          teacherId={1}
        />
      )}
    </>
  );
};

export const Default = Template.bind({});

Default.args = {
  solution: [9, 3, 8, 7, 4, 6, 7, 4, 6],
  playerTotalPoints: 100,
  wrongRepeatPunishment: -50,
  correctRepeatReward: 100,
};
