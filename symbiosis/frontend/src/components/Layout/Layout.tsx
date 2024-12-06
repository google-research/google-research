import SideMenu from "@/components/SideMenu";
import scss from "./Layout.module.scss";
import { useSession } from "next-auth/react";
import React from "react";
import Head from "next/head";
import Footer from "@/components/Footer";

// Layout component for wrapping the application content
const Layout = (props: any) => {
  const { data: session } = useSession(); // Get the user session

  return (
      <>
      <Head>
        <title>Symbiosis</title>
        <meta name="description" content="Symbiosis" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main
          className={scss.layout}
      // Conditionally add padding to the main element if the user is logged in
      style={{ padding: session ? "0 24px 0 80px" : 0 }}
      >
            {session && <SideMenu />} {/* Render the SideMenu if the user is logged in */}
            {props.children} {/* Render the children passed to the Layout component */}
      </main>
</>
);
};

export default Layout;