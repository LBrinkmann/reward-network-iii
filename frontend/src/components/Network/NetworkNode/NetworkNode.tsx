import React, { useEffect, useState } from "react";

import NetworkNodeStyled from "./NetworkNode.styled";
import TutorialTip from "../../Tutorial/TutorialTip";

export type NetworkNodeStatus =
  | "normal"
  | "starting"
  | "active"
  | "disabled"
  | "next";

export interface NetworkNodeInterface {
  /** Node index */
  nodeInx: number;
  /** Text inside the node */
  Text: string;
  /** Node radius, fetched from backend */
  Radius: number;
  /** Stroke color */
  strokeColor?: string;
  /** Node x position */
  x: number;
  /** Node y position */
  y: number;
  /** Callback to handle node click */
  onNodeClick: (nodeIdx: number) => void;
  isValidMove: boolean;
  status: NetworkNodeStatus;
  /** show tutorial tip */
  showTutorial?: boolean;
  /** Callback to handle tutorial tip close */
  onTutorialClose?: () => void;
  nextNodeColor?: string;
}

const NetworkNode: React.FC<NetworkNodeInterface> = (props) => {
  const { showTutorial = false, strokeColor = "black" } = props;
  const [wrongClick, setWrongClick] = useState(false);

  useEffect(() => {
    if (wrongClick) {
      // set timeout to reset status
      setTimeout(() => {
        setWrongClick(false);
      }, 400);
    }
  }, [wrongClick]);

  const nodeClickHandler = () => {
    props.onNodeClick(props.nodeInx);
    if (props.status === "normal" && !props.isValidMove) {
      setWrongClick(true);
    }
  };

  const tutorialId =
    props.nodeInx === 4 ? "practice_multi_edge" : "practice_node";

  return (
    <TutorialTip
      tutorialId={tutorialId}
      isTutorial={showTutorial}
      isShowTip={false}
      onTutorialClose={props.onTutorialClose}
      placement="left"
    >
      <NetworkNodeStyled
        status={props.status}
        fontSize={props.Radius*1.2}
        onClick={nodeClickHandler}
        wrongClick={wrongClick}
        nextNodeColor={props.nextNodeColor}
      >
        <circle
          cx={props.x}
          cy={props.y}
          r={props.Radius}
          key={"circle"}
          stroke={strokeColor}
        />
        <text
          x={props.x}
          y={props.y + props.Radius * 0.40}
          textAnchor="middle"
          key={"state-name"}
        >
          {props.Text.slice(0, 1)}
        </text>
      </NetworkNodeStyled>
    </TutorialTip>
  );
};

export default NetworkNode;
