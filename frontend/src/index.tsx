import React from "react";
import ReactDOM from "react-dom";

import App from "./components/App";
import NetworkSL from "./components/Streamlit/NetworkSL";
import { BrowserRouter, Route, Routes } from "react-router-dom";

ReactDOM.render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />}></Route>
      <Route path="/streamlit" element={<NetworkSL />}></Route>
    </Routes>
  </BrowserRouter>,
  document.querySelector("#root")
);
