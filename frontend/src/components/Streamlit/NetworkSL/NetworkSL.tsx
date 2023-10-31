import React, { FC, useEffect } from "react";
import { Box, Grid } from "@mui/material";
import StaticNetwork, {
  StaticNetworkEdgeInterface,
  StaticNetworkNodeInterface,
} from "../../Network/StaticNetwork/StaticNetwork";
import { selectPossibleMoves } from "../../../reducers/NetworkReducer";

interface NetworkSLInterface {
  network:
    | {
        edges: StaticNetworkEdgeInterface[];
        nodes: StaticNetworkNodeInterface[];
        starting_node: number;
      }
    | undefined;
  maxMoves: number;
  showAllEdges: boolean;
}

const NetworkSL: FC<NetworkSLInterface> = (props) => {
  const { network, maxMoves, showAllEdges } = props;
  const [moves, setCurrentMoves] = React.useState([network.starting_node]);
  const [totalPoints, setTotalPoints] = React.useState(0);
  const [possibleMoves, setPossibleMoves] = React.useState<number[]>([]);
  const [displayEdges, setDisplayEdges] = React.useState<
    StaticNetworkEdgeInterface[]
  >([]);

  useEffect(() => {
    setPossibleMoves(
      selectPossibleMoves(network.edges, moves[moves.length - 1])
    );
  }, [moves, network]);

  useEffect(() => {
    if (!showAllEdges) {
      setDisplayEdges(
        network.edges.filter((edge: StaticNetworkEdgeInterface) =>
          moves.includes(edge.source_num)
        )
      );
    } else {
      setDisplayEdges(network.edges);
    }
  }, [moves, network, showAllEdges]);

  const NodeClickHandler = (nodeIdx: number) => {
    if (possibleMoves.includes(nodeIdx) && moves.length <= maxMoves) {
      const currentEdge = network.edges.filter(
        (edge: any) =>
          edge.source_num === moves[moves.length - 1] &&
          edge.target_num === nodeIdx
      )[0];
      setTotalPoints(totalPoints + currentEdge.reward);
      setCurrentMoves(moves.concat([nodeIdx]));
    }
  };

  return (
    <Grid
      container
      sx={{ margin: "auto", width: "85%" }}
      justifyContent="space-around"
    >
      <StaticNetwork
        edges={displayEdges}
        possibleMoves={possibleMoves}
        nodes={network.nodes}
        currentNodeId={moves[moves.length - 1]}
        onNodeClickHandler={NodeClickHandler}
      />
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Box sx={{ fontSize: "1.2rem", fontWeight: "bold" }}>Total Points</Box>
        <Box sx={{ fontSize: "1.5rem", fontWeight: "bold" }}>{totalPoints}</Box>
        <Box sx={{ fontSize: "1.2rem", fontWeight: "bold" }}>Moves</Box>
        <Box sx={{ fontSize: "1.5rem", fontWeight: "bold" }}>
          {moves.length - 1}
        </Box>
      </Box>
    </Grid>
  );
};

export { NetworkSL, NetworkSLInterface };
