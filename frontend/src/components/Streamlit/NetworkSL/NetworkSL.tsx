import React, { FC, useRef } from "react";
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
  const { network, showAllEdges, trial_name, moves } = props;
  const networkSvgs: React.RefObject<SVGSVGElement>[] = Array.from(
    { length: 11 },
    () => useRef<SVGSVGElement>(null)
  );
  const linSvgs: React.RefObject<SVGSVGElement>[] = Array.from(
    { length: 11 },
    () => useRef<SVGSVGElement>(null)
  );

  if (!network || !moves) return null;

  const idx = moves.length - 1; // Get the index for the last iteration
  const sel_moves = moves.slice(0, idx + 1);
  const possibleMoves = selectPossibleMoves(
    network.edges,
    sel_moves[sel_moves.length - 1]
  );
  const totalPoints = sel_moves
    .slice(1)
    .reduce((acc: number, move: number, idx: number) => {
      const edge = network.edges.find(
        (edge: StaticNetworkEdgeInterface) =>
          edge.source_num === sel_moves[idx] && edge.target_num === move
      );
      return acc + (edge ? edge.reward : 0);
    }, 0);
  const displayEdges = showAllEdges
    ? network.edges
    : network.edges.filter((edge: StaticNetworkEdgeInterface) =>
        sel_moves.includes(edge.source_num)
      );

  const downloadMergedSvg = (svgRefs, name: string) => {
    // Create a new SVG element as a container
    const containerSvg = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "svg"
    );
    let totalWidth = 0,
      maxHeight = 0;
    const margin = 10; // Space between SVGs

    svgRefs.forEach((svgRef) => {
      if (!svgRef.current) return;
      const clone = svgRef.current.cloneNode(true);

      // Get and apply computed styles for each SVG
      const styleSheets = Array.from(document.styleSheets);
      const cssRules = [];
      styleSheets.forEach((sheet) => {
        try {
          if (sheet.cssRules) {
            cssRules.push(...Array.from(sheet.cssRules));
          }
        } catch (e) {
          console.warn("Cannot access stylesheet: " + sheet.href);
        }
      });

      const relevantCss = cssRules
        .filter((rule) => clone.querySelector(rule.selectorText))
        .map((rule) => `${rule.selectorText} { ${rule.style.cssText} }`)
        .join("\n");

      const styleElem = document.createElement("style");
      styleElem.setAttribute("type", "text/css");
      styleElem.innerHTML = relevantCss;
      clone.insertBefore(styleElem, clone.firstChild);

      // Adjust the position of each SVG and append it to the container
      let bbox = svgRef.current.getBBox();
      clone.setAttribute(
        "transform",
        `translate(${totalWidth + margin - 10}, -10)`
      );
      containerSvg.appendChild(clone);

      totalWidth += bbox.width + margin;
      maxHeight = Math.max(maxHeight, bbox.height);
    });

    containerSvg.setAttribute("width", totalWidth + margin);
    containerSvg.setAttribute("height", maxHeight + margin);

    // Serialize the container SVG to a string
    const serializer = new XMLSerializer();
    let htmlStr = serializer.serializeToString(containerSvg);

    htmlStr = htmlStr.replace(/&gt;/g, ">");
    htmlStr = htmlStr.replace(/&lt;/g, "<");
    htmlStr = htmlStr.replace(
      /<style xmlns="http:\/\/www.w3.org\/1999\/xhtml"/g,
      "<style"
    );

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
      spacing={2}
      sx={{ margin: "auto", width: "85%" }}
      direction="column"
    >
      <Grid item sx={{ width: `${100 / moves.length}%` }}>
        <LinearSolution
          edges={network.edges}
          nodes={network.nodes}
          moves={sel_moves}
          svgRef={linSvgs[idx]}
        />
        <StaticNetwork
          edges={displayEdges}
          possibleMoves={possibleMoves}
          nodes={network.nodes}
          currentNodeId={sel_moves[sel_moves.length - 1]}
          svgRef={networkSvgs[idx]}
        />
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <Box sx={{ fontSize: "1.5rem", fontWeight: "bold" }}>
            {totalPoints}
          </Box>
        </Box>
      </Grid>
      <Grid item sx={{ width: "100%" }}>
        <button
          onClick={() =>
            downloadMergedSvg(
              networkSvgs,
              `network_${trial_name}_${moves.length}`
            )
          }
        >
          Download Network
        </button>
        <button
          onClick={() =>
            downloadMergedSvg(linSvgs, `linsol ${trial_name}_${moves.length}`)
          }
        >
          Download LinSol
        </button>
      </Grid>
    </Grid>
  );
};

export { NetworkSL, NetworkSLInterface };
