import { signIn, signOut, useSession } from "next-auth/react";
import Button from "@mui/material/Button";

// Login component for handling user authentication
const Login = () => {
  const { data: session } = useSession(); // Get the user session

  // If the user is logged in, render the "Sign out" button
  if (session) {
    return (
        <>
          <Button variant="contained" color="error" onClick={() => signOut()}>
            Sign out
          </Button>
        </>
    );
  }

  // If the user is not logged in, render the "Sign in" button
  return (
      <>
        <Button variant="contained" color="success" onClick={() => signIn()}>
          Sign in
        </Button>
      </>
  );
};

export default Login;