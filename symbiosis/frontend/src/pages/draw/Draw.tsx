// Draw.tsx
"use client";
import React, {useState, useRef} from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import DrawFlowComponent from '@/components/Draw/DrawFlowComponent';
import Container from '@mui/material/Container';
import Divider from '@mui/material/Divider';
import Button from '@mui/material/Button';
import {TextField} from '@mui/material';
import IconButton from '@mui/material/IconButton';
import SendIcon from '@mui/icons-material/Send';
import {useTheme} from '@mui/material/styles';

const Draw: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const theme = useTheme();

  const handleCardClick = (cardPrompt: string) => {
    setPrompt(cardPrompt); // Update the prompt state
  };

  const prompt1 = "Climate change";
  const prompt2 = "Algorithmic Bias";
  const prompt3 = "Causal factors preventing UN SDG progress on Poverty"

  const inputRef = useRef<HTMLInputElement>(null);

  return (
      <Container maxWidth="lg">
        <Box className="create-main"
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',  // Horizontally center content
              justifyContent: 'center', // Vertically center content
              minHeight: '100vh',
              padding: '2rem 0'
            }}
        >
          {prompt ? ( // Conditionally render based on prompt
              <DrawFlowComponent prompt={prompt}/>
          ) : (
              <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'flex-start',
                    minHeight: '100vh',
                    padding: '2rem 0' // Add some vertical padding
                  }}
              >
                <Typography
                    variant="h2"
                    component="h1"
                    sx={{
                      marginBottom: '2rem',
                      fontFamily: "'Roboto', sans-serif",
                      fontSize: {xs: '1.2rem', sm: '1.4rem', md: '1.7rem'}, // Responsive font size
                      color: 'white',
                      maxWidth: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      lineHeight: 1.6,
                      textAlign: 'center',
                    }}
                >
                  Leverage &nbsp;
                  <img
                      src='/Gemini_RGB.png'
                      alt='Gemini'
                      style={{
                        maxHeight: "1.1em",
                        width: "auto",
                      }}
                  />
                  &nbsp; to create and explore systems thinking models.
                </Typography>
                <Typography
                    variant="subtitle1"
                    component="p"
                    sx={{
                      marginBottom: '2rem',
                      fontFamily: "'Roboto', sans-serif",
                      fontSize: {xs: '1rem', sm: '1.2rem', md: '1.5rem'}, // Responsive font size
                      color: '#8a858e',
                      maxWidth: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      lineHeight: 1.6,
                      textAlign: 'center',
                    }}
                >
                  Start with these sample prompts
                </Typography>
                <Box>
                  <Box sx={{display: 'flex', flexDirection: 'column', gap: 2}}>
                    <Box sx={{display: 'flex', gap: 2}}>
                      <Card
                          sx={{
                            minWidth: 275,
                            cursor: 'pointer',
                            boxShadow: 3, // Add a subtle shadow
                            borderRadius: 2, // Round the corners slightly
                            backgroundColor: theme.palette.primary.main, // Use your primary theme color
                            color: theme.palette.primary.contrastText, // Ensure text contrast
                            '&:hover': {
                              boxShadow: 5, // Increase shadow on hover
                              transform: 'translateY(-2px)' // Add a subtle lift on hover
                            }
                          }}
                          onClick={() => handleCardClick(prompt1)}>
                        <CardContent sx={{textAlign: 'center'}}>
                          <Typography variant="h6" component="div">
                            {prompt1}
                          </Typography>
                        </CardContent>
                      </Card>

                      <Card
                          sx={{
                            minWidth: 275,
                            cursor: 'pointer',
                            boxShadow: 3, // Add a subtle shadow
                            borderRadius: 2, // Round the corners slightly
                            backgroundColor: theme.palette.primary.main, // Use your primary theme color
                            color: theme.palette.primary.contrastText, // Ensure text contrast
                            '&:hover': {
                              boxShadow: 5, // Increase shadow on hover
                              transform: 'translateY(-2px)' // Add a subtle lift on hover
                            }
                          }}
                          onClick={() => handleCardClick(prompt2)}>
                        <CardContent sx={{textAlign: 'center'}}>
                          <Typography variant="h6" component="div">
                            {prompt2}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Box>

                    <Card sx={{
                      minWidth: 275,
                      cursor: 'pointer',
                      boxShadow: 3, // Add a subtle shadow
                      borderRadius: 2, // Round the corners slightly
                      backgroundColor: theme.palette.primary.main, // Use your primary theme color
                      color: theme.palette.primary.contrastText, // Ensure text contrast
                      '&:hover': {
                        boxShadow: 5, // Increase shadow on hover
                        transform: 'translateY(-2px)' // Add a subtle lift on hover
                      }
                    }}
                          onClick={() => handleCardClick(prompt3)}>
                      <CardContent sx={{textAlign: 'center'}}>
                        <Typography variant="h6" component="div">
                          {prompt3}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Box>
                </Box>
                <br/>
                <br/>
                <Box sx={{
                  display: 'flex',
                  alignItems: 'center',
                  my: 4, // Add margin top and bottom
                  width: '100%' // Adjust width as needed
                }}>
                  <Divider sx={{
                    flexGrow: 1,
                    borderColor: '#ccc'
                  }}/> {/* Adjust color as needed */}
                  <Typography variant="subtitle1" component="p" sx={{
                    mx: 2,
                    color: '#666'
                  }}> {/* Adjust color as needed */}
                    Or write your own prompt below...
                  </Typography>
                  <Divider sx={{
                    flexGrow: 1,
                    borderColor: '#ccc'
                  }}/> {/* Adjust color as needed */}
                </Box>
                <Box className="create-search" sx={{display: 'flex', gap: 2, mt: 2}}>
                  <TextField
                      label="Enter your prompt"
                      variant="outlined"
                      fullWidth
                      inputRef={inputRef}
                      sx={{
                        input: {color: 'white'},
                        '& .MuiOutlinedInput-root': {
                          '& fieldset': {
                            borderColor: 'white',
                          },
                          '&:hover fieldset': {
                            borderColor: 'white',
                          },
                          '&.Mui-focused fieldset': {
                            borderColor: 'white',
                          },
                        },
                        width: {
                          xs: '100%', // Full width on extra small screens
                          sm: '500px', // 300px wide on small screens and above
                          md: '800px', // 400px wide on medium screens and above
                        }
                      }}
                      InputProps={{
                        endAdornment: (
                            <IconButton
                                type="submit" // This makes the IconButton submit the form
                                color="primary"
                                onClick={() => {
                                  const inputValue = inputRef.current?.value || "";
                                  if (inputValue.trim() !== "") { // Check if the input is not empty after trimming whitespace
                                    handleCardClick(inputValue);
                                  }
                                }}
                            >
                              <SendIcon/>
                            </IconButton>
                        ),
                      }}
                  />
                </Box>
              </Box>
          )}
        </Box>
      </Container>
  );
};

export default Draw;