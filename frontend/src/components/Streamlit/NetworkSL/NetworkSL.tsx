import React, { FC, useEffect, useRef } from "react";
import { Box, Grid } from "@mui/material";
import StaticNetwork, {
  StaticNetworkEdgeInterface,
  StaticNetworkNodeInterface,
} from "../../Network/StaticNetwork/StaticNetwork";
import LinearSolution from "../../Network/LinearSolution";

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
  moves: number[] | undefined;
  trial_name: string;
}

const NetworkSL: FC<NetworkSLInterface> = (props) => {
  const { network, maxMoves, showAllEdges, trial_name } = props;
  const [moves, setCurrentMoves] = React.useState([network.starting_node]);
  const [totalPoints, setTotalPoints] = React.useState(0);
  const [possibleMoves, setPossibleMoves] = React.useState<number[]>([]);
  const [displayEdges, setDisplayEdges] = React.useState<
    StaticNetworkEdgeInterface[]
  >([]);
  const networkSvg = useRef<SVGSVGElement>(null);
  const linSvg = useRef<SVGSVGElement>(null);



  useEffect(() => {
   if (props.moves) {
      setCurrentMoves(props.moves);
    }
  }, [props.moves]);

  useEffect(() => {
    setPossibleMoves(
      selectPossibleMoves(network.edges, moves[moves.length - 1])
    );
    const moveEdges = moves.slice(1).map((move: number, idx: number) => {
      const edge = network.edges.find(
        (edge: StaticNetworkEdgeInterface) => edge.source_num === moves[idx] && edge.target_num === move
      );
      return edge;
    });
    const totalPoints = moveEdges.reduce((acc: number, edge: StaticNetworkEdgeInterface) => acc + edge.reward, 0);
    setTotalPoints(totalPoints);

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
      setCurrentMoves(moves.concat([nodeIdx]));
    }
  };

  const downloadSvg = (svgRef, name: string) => {
    // Clone the SVG element to avoid modifying the original one
    const clone = svgRef.current.cloneNode(true);

    // Get the computed styles of the SVG and its children
    const styleSheets = Array.from(document.styleSheets);
    const cssRules = [];
    styleSheets.forEach(sheet => {
        try {
            if (sheet.cssRules) {
                cssRules.push(...Array.from(sheet.cssRules));
            }
        } catch (e) {
            console.warn('Cannot access stylesheet: ' + sheet.href);
        }
    });

    // Filter rules that are relevant to the SVG and fix special characters
    const relevantCss = cssRules
        .filter(rule => clone.querySelector(rule.selectorText))
        .map(rule => `${rule.selectorText} { ${rule.style.cssText} }`)
        .join('\n');


    console.log(relevantCss);

    // Create a <style> element with the relevant CSS
    const styleElem = document.createElement("style");
    styleElem.setAttribute("type", "text/css");
    styleElem.innerHTML = relevantCss;
    

    console.log('styleElem', styleElem);

    // Prepend the style element to the clone
    clone.insertBefore(styleElem, clone.firstChild);

    // Serialize the cloned SVG to a string
    const serializer = new XMLSerializer();
    let htmlStr = serializer.serializeToString(clone);

    // Replace HTML entities with their original characters
    htmlStr = htmlStr.replace(/&gt;/g, '>');
    htmlStr = htmlStr.replace(/&lt;/g, '<'); // Add more replacements as needed

    // Remove the xmlns attribute from the style tag
    htmlStr = htmlStr.replace(/<style xmlns="http:\/\/www.w3.org\/1999\/xhtml"/g, '<style');


    console.log('htmlStr', htmlStr);

    // Create and trigger a download action as before
    const blob = new Blob([htmlStr], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.setAttribute("download", `${name}.svg`);
    a.setAttribute("href", url);
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
};



  return (
    <Grid
      container
      sx={{ margin: "auto", width: "85%" }}
      justifyContent="space-around"
    >
      <Grid
        item
        sx={{ marginTop: "10px", marginBottom: "0px", width: "100%" }}
      >
      <LinearSolution
        edges={network.edges}
        nodes={network.nodes}
        moves={moves}
        svgRef={linSvg}
      />
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
        <StaticNetwork
          edges={displayEdges}
          possibleMoves={possibleMoves}
          nodes={network.nodes}
          currentNodeId={moves[moves.length - 1]}
          onNodeClickHandler={NodeClickHandler}
          svgRef={networkSvg}
        />
      </Grid>
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
        <Box sx={{ fontSize: "1.2rem", fontWeight: "bold" }}><button onClick={() => downloadSvg(networkSvg, `network_${trial_name}_${moves.length}`)}>Download Network</button></Box>
        <Box sx={{ fontSize: "1.2rem", fontWeight: "bold" }}><button onClick={() => downloadSvg(networkSvg, `lin_${trial_name}_${moves.length}`)}>Download Lin Sol</button></Box>
      </Box>
    </Grid>
  );
};

export { NetworkSL, NetworkSLInterface };
