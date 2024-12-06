import {ThemeOptions} from "@mui/material";

const darkTheme: ThemeOptions = {
    palette: {
        mode: 'dark',
        primary: {
            main: '#7952b3',
            contrastText: '#ffffff',
        },
        secondary: {
            main: '#e9ecef',
            contrastText: '#7952b3',
        },
        background: {
            default: "#212529",
            paper: "#343A40"
        }
    }
}

export default darkTheme;