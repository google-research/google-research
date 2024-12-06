import IconButton from "@mui/material/IconButton";
import Brightness7Icon from "@mui/icons-material/Brightness7";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import React from "react";
import { useTheme } from "@mui/system";
import Typography from "@mui/material/Typography";
import { useMediaQuery } from "@mui/material";

// Interface for the props of the ThemeToggleButton component
export type ThemeToggleButtonProps = {
  ColorModeContext: React.Context<{ toggleColorMode: () => void }>;
};

const ThemeToggleButton = (props: ThemeToggleButtonProps) => {
  const mobileCheck = useMediaQuery("(min-width: 500px)"); // Check if the screen is mobile
  const { ColorModeContext = React.createContext({ toggleColorMode: () => {} }) } = props; // Get the ColorModeContext from props or create a default one
  const theme = useTheme(); // Access the theme to get the current color mode
  const colorMode = React.useContext(ColorModeContext); // Get the color mode context

  return (
      <>
        {/* Conditionally render the current theme mode text if the screen is not mobile */}
        {mobileCheck && (
            <Typography>{theme.palette.mode}</Typography>
        )}
        {/* IconButton for toggling the theme mode */}
        <IconButton
            sx={{ mr: 2 }}
            title={theme.palette.mode + " mode"} // Set the title attribute for accessibility
            aria-label={theme.palette.mode + " mode button"} // Set the aria-label for accessibility
            onClick={colorMode.toggleColorMode} // Call the toggleColorMode function from the context
            color="inherit"
        >
          {/* Conditionally render the appropriate icon based on the current theme mode */}
          {theme.palette.mode === "dark" ? <Brightness7Icon /> : <Brightness4Icon />}
        </IconButton>
      </>
  );
};

export default ThemeToggleButton;