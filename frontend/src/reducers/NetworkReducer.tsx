import {
  StaticNetworkEdgeInterface,
  StaticNetworkInterface,
  StaticNetworkNodeInterface,
} from "../components/Network/StaticNetwork/StaticNetwork";
import { NetworkEdgeStyle } from "../components/Network/NetworkEdge/NetworkEdge";
import { networkInitialState, NetworkState } from "../contexts/NetworkContext";

export const NETWORK_ACTIONS = {
  SET_NETWORK: "setNetwork",
  NEXT_NODE: "nextNode",
  TIMER_UPDATE: "timeUpdate",
  DISABLE: "disable",
  NEXT_TUTORIAL_STEP: "nextTutorialStep",
  FINISH_COMMENT_TUTORIAL: "finishCommentTutorial",
  HIGHLIGHT_EDGE_TO_CHOOSE: "highlightEdgeToRepeat",
  RESET_EDGE_STYLES: "resetEdgeStyles",
};

const nextTutorialStepReducer = (state: NetworkState, action: any) => {
  if (!state.isPractice) return state;

  if (state.tutorialOptions.start) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        node: true,
      },
    };
  }

  if (state.tutorialOptions.node && state.moves.length < 7) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        general_edge: true,
      },
    };
  } else if (state.tutorialOptions.node && state.moves.length >= 7) {
    return {
      ...state,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        linearSolution: true,
      },
    };
  }

  if (state.tutorialOptions.general_edge) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        edge: true,
      },
    };
  }

  if (state.tutorialOptions.edge) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        general_points: true,
      },
    };
  }

  if (state.tutorialOptions.general_points) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        linearSolution: true,
      },
    };
  }

  if (state.tutorialOptions.linearSolution && state.moves.length === 9) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        time: true,
      },
    };
  } else if (state.tutorialOptions.linearSolution && state.moves.length === 7) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        node: true,
      },
    };
  } else if (state.tutorialOptions.linearSolution && state.moves.length < 9) {
    return {
      ...state,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        linearSolution: true,
      },
    };
  }

  if (state.tutorialOptions.time) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        points: true,
      },
    };
  }

  if (state.tutorialOptions.points) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: {
        ...networkInitialState.tutorialOptions,
        totalScore: true,
      },
    };
  }

  // final step
  if (state.tutorialOptions.totalScore) {
    return {
      ...state,
      tutorialStep: state.tutorialStep + 1,
      tutorialOptions: networkInitialState.tutorialOptions,
    };
  }

  return {
    ...state,
    // clear tutorial options
    tutorialOptions: networkInitialState.tutorialOptions,
  };
};

const timerUpdateReducer = (state: NetworkState, action: any) => {
  // if timer is done
  if (action.payload.time === state.timer.timePassed) {
    return {
      ...state,
      isNetworkDisabled: true,
      isNetworkFinished: true,
      currentNode: undefined,
      timer: {
        ...state.timer,
        isTimerDone: true,
      },
    };
  }
  // if timer is paused
  if (action.payload.paused)
    return {
      ...state,
      timer: {
        ...state.timer,
        isTimerPaused: true,
      },
    };

  // if timer is not done
  return {
    ...state,
    timer: {
      ...state.timer,
      timePassed: state.timer.timePassed + 1,
    },
  };
};

const setNetworkReducer = (state: NetworkState, action: any) => {
  const { edges, nodes } = action.payload.network;
  const startNode = nodes.filter(
    (node: StaticNetworkNodeInterface) => node.starting_node
  )[0].node_num;
  const possibleMoves = selectPossibleMoves(edges, startNode);
  const allowedMoves =
    action.payload.trialType === "repeat"
      ? [action.payload.solution.moves[1]]
      : possibleMoves;

  const animatedMoves =
    action.payload.trialType === "demonstration"
      ? [action.payload.solution.moves[1]]
      : [];
  const highlightedMoves = possibleMoves.filter(
    (move) => !animatedMoves.includes(move)
  );

  return {
    ...networkInitialState,
    network: highlightEdges(action.payload.network, startNode, {
      animated: animatedMoves,
      highlighted: highlightedMoves,
    }),
    currentNode: startNode,
    possibleMoves: possibleMoves,
    allowedMoves: allowedMoves,
    moves: [startNode],
    isNetworkDisabled: false,
    isNetworkFinished: false,
    // Tutorial
    isPractice: action.payload.isPractice,
    tutorialStep: action.payload.isPractice
      ? 1
      : networkInitialState.tutorialStep,
    tutorialOptions: {
      ...networkInitialState.tutorialOptions,
      start: action.payload.isPractice,
      comment: action.payload.commentTutorial,
    },
    teacherComment: action.payload.teacherComment,
    solution: action.payload.solution,
    wrongRepeatPunishment: action.payload.wrongRepeatPunishment,
    correctRepeatReward: action.payload.correctRepeatReward,
    trialType: action.payload.trialType,
  };
};

const nextNodeReducer = (state: NetworkState, action: any) => {
  // if network is disabled or finished, do nothing
  if (state.isNetworkFinished || state.isNetworkDisabled) return state;

  const nextNode = action.payload.nodeIdx;
  const maxStep = action.payload?.maxSteps || 10;

  // if node is not in possible moves, do nothing
  if (!state.possibleMoves.includes(nextNode)) return state;

  // find the current edge
  const currentEdges = state.network.edges.filter(
    (edge: any) =>
      edge.source_num === state.currentNode && edge.target_num === nextNode
  );

  // if edge is not found, do nothing and return state
  if (currentEdges.length !== 1) return state;

  const currentEdge = currentEdges[0];

  // if edge is undefined, do nothing and return state
  if (
    !currentEdge ||
    typeof currentEdge.reward === "undefined" ||
    typeof currentEdge === "undefined"
  )
    return state;

  if (state.trialType === "repeat") {
    const nextSolutionNode = state.solution.moves[state.step + 1];
    if (nextSolutionNode === nextNode) {
      const possibleMoves = selectPossibleMoves(
        state.network.edges,
        nextSolutionNode
      );
      return {
        ...state,
        network: highlightEdges(state.network, nextNode, {
          highlighted: possibleMoves,
        }),
        currentNode: nextNode,
        wrongRepeat: false,
        moves: state.moves.concat([nextNode]),
        correctRepeats: state.correctRepeats.concat([!state.wrongRepeat]),
        points: state.wrongRepeat
          ? state.points
          : state.points + state.correctRepeatReward,
        currentReward: state.wrongRepeat
          ? state.currentReward
          : state.correctRepeatReward,
        rewardIdx: state.wrongRepeat ? state.rewardIdx : state.rewardIdx + 1,
        step: state.step + 1,
        allowedMoves: [state.solution.moves[state.step + 2]],
        possibleMoves: possibleMoves,
        isNetworkDisabled: state.step + 1 >= maxStep,
        isNetworkFinished: state.step + 1 >= maxStep,
      };
    }
    if (state.wrongRepeat) return state;
    return {
      ...state,
      network: highlightEdges(state.network, state.currentNode, {
        dashed: [nextSolutionNode],
      }),
      wrongRepeat: true,
      points: state.points + state.wrongRepeatPunishment,
      currentReward: state.wrongRepeatPunishment,
      rewardIdx: state.rewardIdx + 1,
    };
  }

  const possibleMoves =
    state.trialType === "demonstration"
      ? [action.nextMove]
      : selectPossibleMoves(state.network.edges, nextNode);
  const collectReward = state.trialType === "demonstration" ? false : true;

  const animatedMoves =
    state.trialType === "demonstration"
      ? [state.solution.moves[state.step + 2]]
      : [];
  const highlightedMoves = possibleMoves.filter(
    (move) => !animatedMoves.includes(move)
  );

  return {
    ...state,
    network: highlightEdges(state.network, nextNode, {
      animated: animatedMoves,
      dashed: highlightedMoves,
    }),
    currentNode: nextNode,
    moves: state.moves.concat([nextNode]),
    points: collectReward ? state.points + currentEdge.reward : state.points,
    currentReward: collectReward ? currentEdge.reward : state.currentReward,
    rewardIdx: collectReward ? state.rewardIdx + 1 : state.rewardIdx,
    step: state.step + 1,
    allowedMoves: possibleMoves,
    possibleMoves,
    isNetworkDisabled: state.step + 1 >= maxStep,
    isNetworkFinished: state.step + 1 >= maxStep,
  };
};

const highlightEdges = (
  network: {
    nodes: StaticNetworkNodeInterface[];
    edges: StaticNetworkEdgeInterface[];
  },
  sourceNode: number,
  styleTargets: { [key: string]: number[] }
) => {
  const edges = network.edges.map((edge: StaticNetworkEdgeInterface) => {
    if (edge.source_num === sourceNode)
      for (const style in styleTargets) {
        if (styleTargets[style].includes(edge.target_num)) {
          edge.edgeStyle = style as NetworkEdgeStyle;
          return edge;
        }
      }
    edge.edgeStyle = "normal";
    return edge;
  });
  return { ...network, edges };
};

const networkReducer = (state: NetworkState, action: any) => {
  console.log(state);
  switch (action.type) {
    case NETWORK_ACTIONS.SET_NETWORK:
      return setNetworkReducer(state, action);
    case NETWORK_ACTIONS.TIMER_UPDATE:
      return timerUpdateReducer(state, action);
    case NETWORK_ACTIONS.NEXT_NODE:
      return nextNodeReducer(state, action);
    case NETWORK_ACTIONS.NEXT_TUTORIAL_STEP:
      return nextTutorialStepReducer(state, action);
    case NETWORK_ACTIONS.FINISH_COMMENT_TUTORIAL:
      return {
        ...state,
        tutorialOptions: { ...state.tutorialOptions, comment: false },
      };
    case NETWORK_ACTIONS.DISABLE:
      return {
        ...state,
        isNetworkDisabled: true,
      };
    default:
      return state;
  }
};

export const selectPossibleMoves = (
  allEdges: StaticNetworkEdgeInterface[],
  currentNodeId: number
) => {
  return allEdges
    .filter((edge) => edge.source_num === currentNodeId)
    .map((edge) => edge.target_num);
};

export default networkReducer;
