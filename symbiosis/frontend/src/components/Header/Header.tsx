import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import Menu from "@mui/material/Menu";
import Container from "@mui/material/Container";
import Avatar from "@mui/material/Avatar";
import Tooltip from "@mui/material/Tooltip";
import MenuItem from "@mui/material/MenuItem";
import { signIn, signOut, useSession } from "next-auth/react";
import ThemeToggleButton from "@/components/ThemeToggleButton";
import { useMediaQuery } from "@mui/material";
import { useTheme } from "@mui/system";
import AllInclusiveIcon from "@mui/icons-material/AllInclusive";

// Interface for the props of the Header component
export type HeaderProps = {
  ColorModeContext: React.Context<{ toggleColorMode: () => void }>;
};

const Header = (props: HeaderProps) => {
  const { ColorModeContext } = props; // Get the ColorModeContext from props
  const { data: session } = useSession(); // Get the user session
  const theme = useTheme(); // Access the theme to get dynamic colors
  const userProfileImg = session?.user?.image as string; // Get the user profile image

  // State variables for controlling the menu anchors
  const [anchorElNav, setAnchorElNav] = React.useState<null | HTMLElement>(null);
  const [anchorElUser, setAnchorElUser] = React.useState<null | HTMLElement>(null);

  // Functions to handle opening and closing the navigation menu
  const handleOpenNavMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorElNav(event.currentTarget);
  };
  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };

  // Functions to handle opening and closing the user menu
  const handleOpenUserMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorElUser(event.currentTarget);
  };
  const handleCloseUserMenu = () => {
    setAnchorElUser(null);
  };

  const tabletCheck = useMediaQuery("(min-width: 768px)"); // Check if the screen is tablet or larger

  return (
      <AppBar position="static" sx={{ marginBottom: "40px" }}>
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            {/* Icon for larger screens */}
            <AllInclusiveIcon sx={{ display: { xs: "none", md: "flex" }, mr: 1 }} />
            {/* Title for larger screens */}
            <Typography
                variant="h6"
                noWrap
                component="a"
                href="/"
                sx={{
                  mr: 2,
                  display: { xs: "none", md: "flex" },
                  fontFamily: "monospace",
                  fontWeight: 700,
                  letterSpacing: ".3rem",
                  color: "inherit",
                  textDecoration: "none",
                }}
            >
              Symbiosis
            </Typography>

            {/* Icon for smaller screens */}
            <AllInclusiveIcon sx={{ display: { xs: "flex", md: "none" }, mr: 1 }} />
            {/* Title for smaller screens */}
            <Typography
                variant="h5"
                noWrap
                component="a"
                href=""
                sx={{
                  mr: 2,
                  display: { xs: "flex", md: "none" },
                  flexGrow: 1,
                  fontFamily: "monospace",
                  fontWeight: 700,
                  letterSpacing: ".3rem",
                  color: "inherit",
                  textDecoration: "none",
                }}
            >
              Symbiosis
            </Typography>

            {/* Display signed-in email on tablet and larger screens */}
            {tabletCheck && (
                <Box sx={{ paddingRight: 5, marginLeft: "auto" }}>
                  <Typography>Signed in as {session?.user?.email}</Typography>
                </Box>
            )}

            <ThemeToggleButton ColorModeContext={ColorModeContext} /> {/* Theme toggle button */}

            {/* User menu */}
            <Box sx={{ flexGrow: 0 }}>
              <Tooltip title="Open profile settings">
                <IconButton onClick={handleOpenUserMenu} sx={{ p: 0 }}>
                  <Avatar alt={session?.user?.name as string} src={userProfileImg} />
                </IconButton>
              </Tooltip>
              <Menu
                  sx={{ mt: "45px" }}
                  id="menu-appbar"
                  anchorEl={anchorElUser}
                  anchorOrigin={{
                    vertical: "top",
                    horizontal: "right",
                  }}
                  keepMounted
                  transformOrigin={{
                    vertical: "top",
                    horizontal: "right",
                  }}
                  open={Boolean(anchorElUser)}
                  onClose={handleCloseUserMenu}
              >
                {/* Login/Logout menu item */}
                <MenuItem onClick={() => (session ? signOut() : signIn())}>
                  <Typography textAlign="center">
                    {session ? "Logout" : "Login"}
                  </Typography>
                </MenuItem>
              </Menu>
            </Box>
          </Toolbar>
        </Container>
      </AppBar>
  );
};

export default Header;