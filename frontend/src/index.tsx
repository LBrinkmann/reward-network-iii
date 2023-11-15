import React from "react";
import ReactDOM from "react-dom";

import App from "./components/App";
import NetworkSLConnection from "./components/Streamlit/NetworkSL";
import { BrowserRouter, Route, Routes, useParams } from "react-router-dom";

const  AppWrapper = () => {
  let { experimentType } = useParams();
  return <App experimentType={experimentType} />;
}



ReactDOM.render(
  <BrowserRouter>
    <Routes>
      <Route path="/:experimentType" element={<AppWrapper />}></Route>
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
