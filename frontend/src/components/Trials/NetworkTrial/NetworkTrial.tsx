import React, { FC, useEffect } from "react";
import useNetworkContext from "../../../contexts/NetworkContext";
import { Box, Divider, Grid, Typography } from "@mui/material";
import StaticNetwork, {
  StaticNetworkEdgeInterface,
} from "../../Network/StaticNetwork/StaticNetwork";
import PlayerInformation from "../PlayerInformation";
import LinearSolution from "../../Network/LinearSolution";
import Timer from "../../Timer";
import { NETWORK_ACTIONS } from "../../../reducers/NetworkReducer";
import Legend from "./RewardsLegend";

interface NetworkTrialInterface {
  showLegend?: boolean;
  showComment?: boolean;
  teacherId?: number;
  showLinearNetwork?: boolean;
  showTimer?: boolean;
  time?: number;
  isPractice?: boolean;
  isTimerPaused?: boolean;
  playerTotalPoints?: number;
  showCurrentNetworkPoints?: boolean;
  showTotalPoints?: boolean;
  allowNodeClick?: boolean;
}

const NetworkTrial: FC<NetworkTrialInterface> = (props) => {
  const {
    showLegend = true,
    showComment = false,
    teacherId = 1,
    showLinearNetwork = true,
    showTimer = true,
    time = 35,
    isPractice = false,
    isTimerPaused = false,
    playerTotalPoints = 0,
    showCurrentNetworkPoints = true,
    showTotalPoints = true,
    allowNodeClick = true,
  } = props;
  const { networkState, networkDispatcher } = useNetworkContext();

  const NodeClickHandler = (nodeIdx: number) => {
    if (!allowNodeClick) return;

    // skip update if network is disabled or finished
    if (networkState.isNetworkDisabled || networkState.isNetworkFinished)
      return;

    // allow clicking only for some tutorial steps
    if (
      isPractice &&
      !(
        networkState.tutorialOptions.edge ||
        networkState.tutorialOptions.linearSolution
      )
    )
      return;

    networkDispatcher({
      type: NETWORK_ACTIONS.NEXT_NODE,
      payload: { nodeIdx },
    });
    if (isPractice)
      networkDispatcher({ type: NETWORK_ACTIONS.NEXT_TUTORIAL_STEP });
  };

  const NextTutorialStepHandler = () =>
    networkDispatcher({ type: NETWORK_ACTIONS.NEXT_TUTORIAL_STEP });

  const onTutorialCommentClose = () =>
    networkDispatcher({ type: NETWORK_ACTIONS.FINISH_COMMENT_TUTORIAL });

  return (
    <Grid
      container
      sx={{ margin: "auto", width: "85%" }}
      justifyContent="space-around"
    >
      <Grid item sx={{ p: 1 }} xs={2}>
        <Grid
          container
          direction="row"
          justifyContent="flex-start"
          style={{ height: "250px" }}
        >
          <Grid item mt={"50px"} xs={12}>
            {showTimer && (
              <Timer
                time={time}
                invisibleTime={5} // 5 seconds before the timer starts
                pause={
                  isTimerPaused ||
                  networkState.isNetworkFinished ||
                  networkState.isNetworkDisabled
                }
                showTutorial={networkState.tutorialOptions.time}
                onTutorialClose={NextTutorialStepHandler}
              />
            )}
          </Grid>
          <Grid item xs={12}>
            <PlayerInformation
              id={teacherId}
              step={networkState.step}
              cumulativePoints={networkState.points}
              totalScore={playerTotalPoints + networkState.points}
              showComment={showComment}
              comment={networkState.teacherComment}
              showTutorialScore={networkState.tutorialOptions.points}
              showTutorialComment={networkState.tutorialOptions.comment}
              onTutorialCommentClose={onTutorialCommentClose}
              showTutorialTotalScore={networkState.tutorialOptions.totalScore}
              onTutorialClose={NextTutorialStepHandler}
              showCumulativePoints={showCurrentNetworkPoints}
              showTotalPoints={showTotalPoints}
            />
          </Grid>
        </Grid>
      </Grid>
      <Grid item xs={6}>
        <Grid
          container
          direction="column"
          justifyContent="space-around"
          alignItems="center"
        >
          <Grid
            item
            sx={{ marginTop: "20px", marginBottom: "50px", width: "100%" }}
          >
            {showLinearNetwork && (
              <LinearSolution
                edges={networkState.network.edges}
                nodes={networkState.network.nodes}
                moves={networkState.moves}
                correctRepeats={networkState.correctRepeats}
                showTutorial={networkState.tutorialOptions.linearSolution}
              />
            )}
          </Grid>
          <Grid
            item
            style={{
              position: "relative",
              width: "100%",
              display: "flex",
              justifyContent: "center",
            }}
          >
            <FlashingReward />
            <StaticNetwork
              edges={networkState.network.edges.filter(
                (edge: StaticNetworkEdgeInterface) =>
                  networkState.moves.includes(edge.source_num)
              )}
              nodes={networkState.network.nodes}
              currentNodeId={
                networkState.isNetworkFinished ? null : networkState.currentNode
              }
              possibleMoves={networkState.possibleMoves}
              allowedMoves={networkState.allowedMoves}
              onNodeClickHandler={NodeClickHandler}
              disableClick={networkState.isNetworkDisabled}
              showEdgeTutorial={networkState.tutorialOptions.edge}
              showNodeTutorial={networkState.tutorialOptions.node}
              onTutorialClose={NextTutorialStepHandler}
              blur={networkState.tutorialOptions.comment}
            />
          </Grid>
          <Grid
            item
            sx={{ marginTop: "20px", marginBottom: "50px", width: "100%" }}
          >
            {networkState.wrongRepeat && (
              <Typography variant="h6" align="center">
                You chose the wrong path. Please select the correct path,
                highlighted here. You will not earn points for this choice.
              </Typography>
            )}
          </Grid>
        </Grid>
      </Grid>
      <Grid item xs={2}>
        <Grid
          container
          direction="column"
          justifyContent="center"
          alignItems="center"
        >
          {showLegend && (
            <Grid item mt={"200px"}>
              <Legend />
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  );
};
const FlashingReward: FC = () => {
  const allRewards = [-50, 0, 100, 200, 400];
  const colors = [
    "#c51b7d",
    "#e9a3c9",
    "#e6f5d0",
    "#a1d76a",
    "#4d9221",
    "#bfbfbf",
  ];
  const { networkState } = useNetworkContext();
  const [show, setShow] = React.useState(false);
  useEffect(() => {
    if (networkState.rewardIdx !== 0) {
      setShow(true);
    }
  }, [networkState.rewardIdx]);

  useEffect(() => {
    if (show) {
      // set timeout to reset status
      setTimeout(() => {
        setShow(false);
      }, 800);
    }
  }, [show]);

  var color;
  var text;
  if (networkState.currentReward === undefined) {
    color = "white";
    text = "";
  } else if (networkState.trialType === "repeat") {
    color = networkState.currentReward > 0 ? colors[5] : colors[5];
    text = networkState.currentReward > 0 ? "Correct" : "Wrong";
  } else {
    color = colors[allRewards.indexOf(networkState.currentReward)];
    text =
      networkState.currentReward > 0
        ? "Gain"
        : networkState.currentReward < 0
        ? "Loss"
        : "Neutral";
  }

  return (
    <Box
      style={{
        position: "absolute", // Position FlashingReward absolutely within the relative Grid item
        top: "50%", // Center vertically
        left: "50%", // Center horizontally
        transform: "translate(-50%, -50%)", // Offset by half the width and height of FlashingReward
        opacity: show ? 0.8 : 0,
      }}
      sx={{
        borderRadius: "50%",
        width: "60px",
        height: "40px",
        bgcolor: color,
        // typography: 'h4',
        // textAlign: 'center',
        px: "40px",
        py: "50px",
        display: "flex", // Use flexbox for alignment
        flexDirection: "column", // Stack child elements vertically
        justifyContent: "center", // Center child elements vertically
        alignItems: "center", // Center child elements horizontally
      }}
    >
      <Typography variant="h4" align="center">
        {text}
      </Typography>
      <Typography variant="h4" align="center">
        {networkState.currentReward}
      </Typography>
    </Box>
  );
};

export default NetworkTrial;
