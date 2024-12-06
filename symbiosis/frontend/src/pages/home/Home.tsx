import * as React from 'react';
import TextField from '@mui/material/TextField';
import Stack from '@mui/material/Stack';
import Autocomplete from '@mui/material/Autocomplete';
import {useState, useEffect} from 'react';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import {useRouter} from 'next/router';
import ExploreMain from '@/components/Explore/ExploreMain';
import {useTour} from '@reactour/tour';

const Home = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const router = useRouter();
  const [tourLoaded, setTourLoaded] = useState(false);

  const {setIsOpen} = useTour();

  useEffect(() => {
    if (!tourLoaded) {
      setIsOpen(true);
      setTourLoaded(true);
    }
  }, []);

  const handleClick = (sdgId: string) => {
    router.push(`/explore/graph?sdgId=${sdgId}`);
  };

  // Fetch data from backend API based on search term
  useEffect(() => {
    const fetchData = async () => {
      if (searchTerm) {
        if (searchTerm.trim() !== '') {
          try {
            const encodedPath = encodeURIComponent(`/search?q=${searchTerm}`);
            const response = await fetch(`$$BACKEND_URL$$?path=${encodedPath}`);
            const data = await response.json();
            console.log(data);

            const options = data.message.map((item) => ({
              label: item.paper_title,
              value: item.paper_title,
              uuid: item.uuid,
              match_field: item.match_field,
              displayLabel: (
                  <Typography component="span">
                    <span style={{fontSize: '1.5em'}}> {' → '}</span>
                    {/*<span style={{ color: '#808080' }}>*/}
                    {/*  Matched paper by { item.match_field === 'title' ? 'title' : 'abstract' }*/}
                    {/*</span>*/}
                    {/*{' → '}*/}
                    <span style={{fontWeight: 'bold', color: '#6495ED'}}>
                      {item.paper_title}
                    </span>
                  </Typography>
              ),
            }));

            setSearchResults(options);
          } catch (error) {
            console.error('Error fetching search results', error);
          }
        } else {
          setSearchResults([]);
        }
      }
    };

    // Debounce to avoid excessive API calls
    const debounceTimer = setTimeout(() => {
      fetchData();
    }, 200);

    return () => clearTimeout(debounceTimer);
  }, [searchTerm]);

  return (
      <Container
          maxWidth="lg"> {/* Use Container for better layout control */}
        <div className="home-main">
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
                  fontWeight: 'bold',
                  marginBottom: '1rem',
                  fontFamily: "'Poppins', sans-serif",
                  fontSize: {xs: '2rem', sm: '3rem', md: '4rem'}, // Responsive font size
                  maxWidth: '80%',
                  textAlign: 'center',
                  color: 'white',
                }}
            >
              Unlock the Power of Systems Thinking with AI
            </Typography>

            <Typography
                variant="subtitle1"
                component="p"
                sx={{
                  marginBottom: '2rem',
                  fontFamily: "'Roboto', sans-serif",
                  fontSize: {xs: '1rem', sm: '1.2rem', md: '1.5rem'}, // Responsive font size
                  color: '#8a858e',
                  maxWidth: '60%',
                  textAlign: 'center',
                  lineHeight: 1.6 // Improve line spacing
                }}
            >
              Dive deep into research papers, visualize complex models, and
              generate new insights with{' '}
              <img
                  src='/Gemini_RGB.png'
                  alt='Gemini'
                  style={{
                    maxHeight: "1.1em",
                    width: "auto",
                    verticalAlign: 'middle'
                  }}
              />
            </Typography>


            <Stack spacing={2} sx={{width: {xs: '90%', sm: '900px'}}}>
                  <Autocomplete
                      freeSolo
                      id="homepage-search"
                      // options={!searchResults ? [{label: "Loading...", id:0}] : searchResults} // Use search results from API
                      options={searchResults}
                      value={searchTerm}
                      onChange={(event, newValue) => {
                        if (newValue) {
                          router.push(`/datatable?uuid=${newValue.uuid}`);
                        }
                      }}
                      onInputChange={(event, newInputValue) => {
                        setSearchTerm(newInputValue);
                      }}
                      renderOption={(props, option) => (
                          <li {...props}>
                            {option.displayLabel}
                          </li>
                      )}
                      groupBy={(option) => `Matches by ${option.match_field === 'title' ? 'Paper title' : 'Paper abstract'}`}
                      renderInput={(params) => (
                          <TextField
                              {...params}
                              label="Type a keyword or query to search for research papers"
                              variant="outlined"
                              sx={{
                                '& .MuiOutlinedInput-root': { // Target the outlined input element
                                  '& fieldset': { // Target the border
                                    borderColor: 'white', // Change border color (optional)
                                  },
                                  '&:hover fieldset': {
                                    borderColor: '#808080', // Change border color on hover (optional)
                                  },
                                  '&.Mui-focused fieldset': {
                                    borderColor: '#6495ED', // Change border color when focused (optional)
                                  },
                                },
                                // '& .MuiInputBase-input': { // Target the actual input field
                                //   backgroundColor: 'lightblue', // Change background color
                                //   color: 'black', // Adjust text color for contrast if needed
                                // }
                              }}
                              slotProps={{
                                input: {
                                  ...params.InputProps,
                                  type: 'search',
                                },
                              }}
                          />
                      )}
                  />
            </Stack>
            <br/>
            <br/>
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              my: 4, // Add margin top and bottom
              width: '80%' // Adjust width as needed
            }}>
              <Divider sx={{
                flexGrow: 1,
                borderColor: '#ccc'
              }}/> {/* Adjust color as needed */}
              <Typography variant="body1" sx={{
                mx: 2,
                color: '#666'
              }}> {/* Adjust color as needed */}
                Or Explore via Sustainable Development Goals (SDGs) below...
              </Typography>
              <Divider sx={{
                flexGrow: 1,
                borderColor: '#ccc'
              }}/> {/* Adjust color as needed */}
            </Box>
            <ExploreMain onButtonClick={handleClick} sdgCount={8}/>
          </Box>
        </div>
      </Container>
  )
      ;
};

export default Home;