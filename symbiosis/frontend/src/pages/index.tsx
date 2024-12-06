import Home from "@/pages/home";
import Login from "@/components/Login";
import { useSession } from "next-auth/react";
import scss from "@/components/Layout/Layout.module.scss";
import React from "react";

const Landing: React.FC = () => {
  const { data: session } = useSession();

  return (
    <main className={scss.main}>
      {session && <Home />}
      {!session && <Login />}
    </main>
  );
};

export default Landing;
