import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import React, { ReactNode } from "react";
import { NetworkSL, NetworkSLInterface } from "./NetworkSL";

interface State {}

/**
 * This is a React-based component template. The `render()` function is called
 * automatically when your component should be re-rendered.
 */
class NetworkSLConnection extends StreamlitComponentBase<State> {
  public render = (): ReactNode => {
    return <NetworkSL {...(this.props.args as NetworkSLInterface)} />;
  };
}

export default withStreamlitConnection(NetworkSLConnection);
