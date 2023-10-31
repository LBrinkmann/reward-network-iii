import {
  StaticNetworkEdgeInterface,
  StaticNetworkNodeInterface,
} from "../components/Network/StaticNetwork/StaticNetwork";
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
  const possibleMoves = action.payload.forceSolution ? [action.payload.solution.moves[1]] : selectPossibleMoves(edges, startNode);

  return {
    ...networkInitialState,
    network: action.payload.network,
    currentNode: startNode,
    possibleMoves: possibleMoves,
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
    forceSolution: action.payload.forceSolution,
  };
};

const nextNodeReducer = (state: NetworkState, action: any) => {
  // if network is disabled or finished, do nothing
  if (state.isNetworkFinished || state.isNetworkDisabled) return state;

  const nextNode = action.payload.nodeIdx;
  const maxStep = action.payload?.maxSteps || 8;

  // if node is not in possible moves, do nothing
  if (!state.possibleMoves.includes(nextNode) && !state.forceSolution) return state;

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

  if (state.forceSolution) {
    const nextSolutionNode = state.solution.moves[state.step + 1];
    const correct = nextSolutionNode === nextNode;
    if (correct) {
      return {
        ...state,
        currentNode: nextSolutionNode,
        moves: state.moves.concat([nextSolutionNode]),
        points: state.points + state.correctRepeatReward,
        currentReward: state.correctRepeatReward,
        rewardIdx: state.rewardIdx + 1,
        step: state.step + 1,
        possibleMoves: [state.solution.moves[state.step + 2]],
        isNetworkDisabled: state.step + 1 >= maxStep,
        isNetworkFinished: state.step + 1 >= maxStep,
      };
    }
    return {
      ...state,
      currentNode: nextSolutionNode,
      moves: state.moves.concat([nextSolutionNode]),
      points: state.points + state.wrongRepeatPunishment,
      currentReward: state.wrongRepeatPunishment,
      rewardIdx: state.rewardIdx + 1,
      step: state.step + 1,
      possibleMoves: [state.solution.moves[state.step + 2]],
      isNetworkDisabled: state.step + 1 >= maxStep,
      isNetworkFinished: state.step + 1 >= maxStep,
    };
    // return {
    //   ...state,
    //   points: state.points + state.wrongRepeatPunishment,
    //   currentReward: state.wrongRepeatPunishment,
    //   rewardIdx: state.rewardIdx + 1,
    // };
  }
  return {
    ...state,
    currentNode: nextNode,
    moves: state.moves.concat([nextNode]),
    points: state.points + currentEdge.reward,
    currentReward: currentEdge.reward,
    rewardIdx: state.rewardIdx + 1,
    step: state.step + 1,
    possibleMoves: selectPossibleMoves(state.network.edges, nextNode),
    isNetworkDisabled: state.step + 1 >= maxStep,
    isNetworkFinished: state.step + 1 >= maxStep,
  };
};

const highlightEdgeToRepeatReducer = (state: NetworkState, action: any) => {
  const { source, target, edgeStyle } = action.payload;
  const edgeToFollow = state.network.edges.filter(
    (edge: StaticNetworkEdgeInterface) =>
      edge.source_num === source && edge.target_num === target
  )[0];

  // set all edges to default style
  state.network.edges.forEach(
    (edge: StaticNetworkEdgeInterface) => (edge.edgeStyle = "normal")
  );

  if (edgeToFollow) {
    edgeToFollow.edgeStyle = edgeStyle;
    return { ...state, possibleMoves: [target] };
  } else return state;
};

const networkReducer = (state: NetworkState, action: any) => {
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
    case NETWORK_ACTIONS.HIGHLIGHT_EDGE_TO_CHOOSE:
      return highlightEdgeToRepeatReducer(state, action);
    case NETWORK_ACTIONS.RESET_EDGE_STYLES:
      const resetEdges = state.network.edges;
      resetEdges.forEach((edge: any) => (edge.edgeStyle = "normal"));
      return { ...state, network: { ...state.network, edges: resetEdges } };
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
