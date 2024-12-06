import {SessionProvider} from "next-auth/react";
import React from "react";
import {useRouter} from "next/router";
import {createTheme, ThemeProvider} from "@mui/material/styles";
import {CssBaseline} from "@mui/material";
import darkTheme from "@/theme/darkTheme";
import lightTheme from "@/theme/lightTheme";
import Header from "@/components/Header";
import Layout from "@/components/Layout";
import './globals.css';
import '@xyflow/react/dist/style.css';
import {TourProvider} from '@reactour/tour';
import {Box, Button, Typography, LinearProgress} from '@mui/material';


const ColorModeContext = React.createContext({
  toggleColorMode: () => {
  },
});

function ContentComponent(props) {
  const isLastStep = props.currentStep === props.steps.length - 1;
  const content = props.steps[props.currentStep].content;

  return (
      <Box
          sx={{
            border: '1px solid #ccc', // Use MUI's styling solution
            padding: 2,
            background: 'white',
            borderRadius: 1,
            boxShadow: 3, // Add a subtle shadow
            position: 'relative', // For positioning the close button
          }}
      >
        {/* Close button */}
        <Button
            onClick={() => props.setIsOpen(false)}
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              minWidth: 0,
              padding: 0.5,
            }}
        >
          x
        </Button>

        {props.currentStep === 0 ? (
            <Typography variant="h6" component="div" sx={{
              mb: 1,
              fontFamily: 'Roboto',
              color: 'black',
              textAlign: 'center',
              fontSize: '1.3rem'
            }}>
              Welcome to Symbiosis <span role="img"
                                         aria-label="waving hand">👋</span>
            </Typography>
        ) : <br/>}

        <Typography component="pre" sx={{
          mb: 1,
          fontFamily: 'Roboto',
          color: '#7952b3',
          textAlign: 'center',
          fontSize: '0.9rem',
          wordBreak: 'break-word'
        }}>
          {/* Check if the step.content is a function or a string */}
          {typeof content === 'function'
              ? content({...props, someOtherStuff: 'Custom Text'})
              : content}
        </Typography>

        <Box sx={{alignItems: 'center', mt: '2'}}>
          <LinearProgress
              variant="determinate"
              value={(props.currentStep + 1) / props.steps.length * 100}
              sx={{width: '100%'}}
          />
          <Typography variant="body2" color="text.secondary">
            {props.currentStep + 1} of {props.steps.length}
          </Typography>
        </Box>

        <Box sx={{display: 'flex', justifyContent: 'space-between', mt: 2}}>
          <Button
              variant="contained"
              size="small"
              color="primary"
              onClick={() => props.setCurrentStep((s) => Math.max(0, s - 1))}
              disabled={props.currentStep === 0}
              sx={{
                "&.Mui-disabled": {
                  background: '#8a6cb6',
                  color: 'white',
                },
                cursor: props.currentStep === 0 ? 'not-allowed' : 'pointer', // Change cursor
              }}
          >
            Back
          </Button>
          <Button
              variant="contained"
              size="small"
              color="primary"
              // disabled={props.currentStep === 2}
              // sx={{
              //   "&.Mui-disabled": {
              //     background: '#8a6cb6',
              //     color: 'white',
              //   },
              //   cursor: props.currentStep === 2 ? 'not-allowed' : 'pointer', // Change cursor
              // }}
              onClick={() => {
                if (isLastStep) {
                  props.setIsOpen(false);
                } else {
                  props.setCurrentStep((s) => s + 1);
                }
              }}
          >
            {isLastStep ? 'Finish' : 'Next'}
          </Button>
        </Box>

        <Button
            onClick={() => props.setIsOpen(false)}
            size="small"
            sx={{
              mt: 1,
              display: 'block', // Make it a block-level element
              mx: 'auto', // Center horizontally
              color: '#666', // Lighter color
              borderColor: '#ddd', // Add a border
              '&:hover': {
                borderColor: '#4a4848', // Slightly darker border on hover
              }
            }}
            variant="outlined" // Add a border
        >
          Skip Tour
        </Button>
      </Box>
  );
}

const steps = [
  {
    selector: ".home-main",
    content:
        `Let's take a tour to get you familiarized
        with Symbiosis. 
        You can skip this tour anytime you want!
        Click Next to continue.`,
    // position: 'center'
  },
  {
    selector: ".home-explore",
    content:
        `This is your main navigation panel.
        
        Clicking this *Explore* button will
        take you to SDG based explorer view. 
        
        Click Next to continue.`,
  },
  {
    selector: ".home-create",
    content:
        `Clicking this *Create* button will take
        you to our GenAI based
        Systems Thinking co-pilot tool.
        
        Let's start here,
        click highlighted "Create" button.
        
        After that click Next to continue tour.`,
    // action:  () => {
    //   return <NavigateToCreate />
    // },
  },
  {
    selector: ".create-main",
    content:
        `Welcome to the Create Model tool.
        
        You can leverage GenAI to create
        systems thinking models based
        on the problem context you provide. 
        
        This tool takes care of translating 
        your problem context to systems thinking
        causal loop diagram notation.
         
        Click Next to continue. 
        `
  },
  {
    selector: ".create-search",
    content:
        `Let's start by providing context
        of your problem or query.
        
        You can start with something simple:
        "How data and historical bias
        impacts healthcare AI systems"
        
        or 
        
        you can simply type a societal
        topic like "Homelessness"
        to start with. 
        
        Click the send button in highlighted
        prompt field after you are done typing!
        
        This is the end of the tour. Click
        Finish to end or Previous to go back.
        `,
  },
];

const App = ({Component, pageProps: {session, ...pageProps}}) => {
  const [mode, setMode] = React.useState<"light" | "dark">("dark");
  const colorMode = React.useMemo(
      () => ({
        toggleColorMode: () => {
          setMode((prevMode) => (prevMode === "light" ? "dark" : "light"));
        },
      }),
      []
  );

  const darkThemeChosen = React.useMemo(
      () =>
          createTheme({
            ...darkTheme,
          }),
      [mode]
  );
  const lightThemeChosen = React.useMemo(
      () =>
          createTheme({
            ...lightTheme,
          }),
      [mode]
  );
  const router = useRouter();
  const [step, setStep] = React.useState(0);

  const setCurrentStep = (step) => {
    switch (step) {
      case 3:
        router.push("/draw");
        break;
      default:
        break;
    }
    setStep(step);
  };

  return (
      <ColorModeContext.Provider value={colorMode}>
        <ThemeProvider
            theme={mode === "dark" ? darkThemeChosen : lightThemeChosen}>
          <TourProvider
              steps={steps}
              currentStep={step}
              setCurrentStep={setCurrentStep}
              ContentComponent={ContentComponent}
              styles={{popover: (base) => ({...base, padding: 0})}}
          >
            <SessionProvider session={session}>
              <CssBaseline/>
              <Header ColorModeContext={ColorModeContext}/>
              <Layout>
                <Component {...pageProps} />
              </Layout>
            </SessionProvider>
          </TourProvider>
        </ThemeProvider>
      </ColorModeContext.Provider>
  );
};
export default App;
