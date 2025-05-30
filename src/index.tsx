import React from "react";
import ReactDOM from "react-dom/client";
import { ChakraProvider } from "@chakra-ui/react";
import FlashcardApp from "./App";
import theme from "./theme";
import './index.css';

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(
  <React.StrictMode>
    <ChakraProvider theme={theme}>
      <FlashcardApp />
    </ChakraProvider>
  </React.StrictMode>
);

