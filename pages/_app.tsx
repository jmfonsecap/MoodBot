import "normalize.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@styles/global.scss";
import type { AppProps } from "next/app";
import { QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { NavigationProvider } from "@features/layout";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <NavigationProvider>
      <Component />
    </NavigationProvider>
  );
}

export default MyApp;
