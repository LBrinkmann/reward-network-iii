import React from "react";
import ReactDOM from "react-dom";

import App from "./components/App";
import NetworkSLConnection from "./components/Streamlit/NetworkSL";
import { BrowserRouter, Route, Routes } from "react-router-dom";

ReactDOM.render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />}></Route>
      <Route
        path="/streamlit"
        element={
          <React.StrictMode>
            <NetworkSLConnection />
          </React.StrictMode>
        }
      ></Route>
    </Routes>
  </BrowserRouter>,
  document.querySelector("#root")
);
