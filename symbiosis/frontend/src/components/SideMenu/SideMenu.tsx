import * as React from "react";
import IconButton from "@mui/material/IconButton";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import Person2Icon from "@mui/icons-material/Person2";
import EqualizerIcon from "@mui/icons-material/Equalizer";
import AddCircleIcon from "@mui/icons-material/AddCircle";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";
import TravelExplore from "@mui/icons-material/TravelExplore";
import { Settings } from "@mui/icons-material";
import NextLink from "next/link";
import scss from "./SideMenu.module.scss";
import HomeIcon from "@mui/icons-material/Home";
import PivotTableChartIcon from '@mui/icons-material/PivotTableChart';

import {
  Divider,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Theme,
  useMediaQuery,
  useTheme,
  CSSObject,
} from "@mui/material";
import { signOut } from "next-auth/react";

const drawerWidth = 240; // Width of the drawer when open

// CSS styles for the drawer when open
const openedMixin = (theme: Theme): CSSObject => ({
  width: drawerWidth,
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.enteringScreen,
  }),
  overflowX: "hidden",
});

// CSS styles for the drawer when closed
const closedMixin = (theme: Theme): CSSObject => ({
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  overflowX: "hidden",
  width: `calc(${theme.spacing(7)} + 1px)`,
  [theme.breakpoints.up("sm")]: {
    width: `calc(${theme.spacing(8)} + 1px)`,
  },
});

// Array of route paths for the menu items
const menuRouteList = [
  "",
  "explore",
  "create",
  "datatable",
];

// Array of display names for the menu items
const menuListTranslations = [
  "Home",
  "Explore",
  "Create",
  "Table View",
];

// Array of icons for the menu items
const menuListIcons = [
  <HomeIcon />,
  <AutoGraphIcon />,
  <AddCircleIcon />,
  <PivotTableChartIcon />,
];

const SideMenu = () => {
  const theme = useTheme(); // Access the theme to get dynamic colors
  const [open, setOpen] = React.useState(false); // State to control the open/closed state of the drawer
  const mobileCheck = useMediaQuery("(min-width: 600px)"); // Check if the screen is mobile

  // Toggle the open/closed state of the drawer
  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  // Handle clicks on list items
  const handleListItemButtonClick = (text: string) => {
    text === "Sign Out" ? signOut() : null; // Sign out if the clicked item is "Sign Out"
    setOpen(false); // Close the drawer on mobile
  };

  // Function to map text to classNames (for applying specific styles)
  const getClassNameForText = (text) => {
    switch (text) {
      case "Home":
        return "";
      case "Explore":
        return "home-explore";
      case "Create":
        return "home-create";
      case "Table View":
        return "home-table";
      default:
        return "";
    }
  };

  return (
      <Drawer
          variant="permanent"
          anchor="left"
          open={open}
          className={scss.sideMenu}
          sx={{
            width: drawerWidth,
            [`& .MuiDrawer-paper`]: {
              left: 0,
              top: mobileCheck ? 64 : 57, // Adjust top position based on screen size
              flexShrink: 0,
              whiteSpace: "nowrap",
              boxSizing: "border-box",
              ...(open && {
                ...openedMixin(theme),
                "& .MuiDrawer-paper": openedMixin(theme),
              }),
              ...(!open && {
                ...closedMixin(theme),
                "& .MuiDrawer-paper": closedMixin(theme),
              }),
            },
          }}
      >
        {/* Drawer header with toggle button */}
        <div className={scss.drawerHeader}>
          <IconButton onClick={handleDrawerToggle}>
            {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
          </IconButton>
        </div>
        <Divider />
        <Divider />
        {/* List of menu items */}
        <List>
          {menuListTranslations.map((text, index) => (
              <ListItem key={text} disablePadding sx={{ display: "block" }}>
                <div className={getClassNameForText(text)}> {/* Apply specific class based on text */}
                  {/* Use NextLink for navigation */}
                  <NextLink
                      className={scss.link}
                      // Determine the href based on the menu item text
                      href={
                        text === "Create"
                            ? `/draw`
                            : text === "Explore"
                                ? `/explore`
                                : text === "Table View"
                                    ? `/datatable`
                                    : `/home`
                      }
                  >
                    <ListItemButton
                        onClick={() => handleListItemButtonClick(text)}
                        title={text}
                        aria-label={text}
                        sx={{
                          minHeight: 48,
                          justifyContent: open ? "initial" : "center",
                          px: 2.5,
                        }}
                    >
                      <ListItemIcon
                          sx={{
                            minWidth: 0,
                            mr: open ? 3 : "auto",
                            justifyContent: "center",
                          }}
                      >
                        {menuListIcons[index]} {/* Render the corresponding icon */}
                      </ListItemIcon>
                      <ListItemText
                          primary={text}
                          sx={{
                            color: theme.palette.text.primary,
                            opacity: open ? 1 : 0, // Hide text when drawer is closed
                          }}
                      />
                    </ListItemButton>
                  </NextLink>
                </div>
              </ListItem>
          ))}
        </List>
      </Drawer>
  );
};

export default SideMenu;