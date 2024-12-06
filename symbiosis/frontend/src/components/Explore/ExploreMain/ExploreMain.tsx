import React from 'react';
import { Box, Grid, Button, Typography } from '@mui/material';
import { useRouter } from 'next/router';
import { useTheme } from '@mui/system';
import Image from 'next/image';

// Define the interface for the props
interface ExploreMainProps {
  onButtonClick: (sdgId: string) => void; // Callback function when an SDG button is clicked
  sdgCount?: number; // Optional number of SDGs to display
}

const ExploreMain: React.FC<ExploreMainProps> = ({ onButtonClick, sdgCount }) => {
  const router = useRouter();
  const theme = useTheme();

  // Array of SDG goals with their IDs and image paths
  const sdgGoals = [
    { id: "SDG01", image: "/Goal-01.png" },
    { id: "SDG02", image: "/Goal-02.png" },
    { id: "SDG03", image: "/Goal-03.png" },
    { id: "SDG04", image: "/Goal-04.png" },
    { id: "SDG05", image: "/Goal-05.png" },
    { id: "SDG06", image: "/Goal-06.png" },
    { id: "SDG07", image: "/Goal-07.png" },
    { id: "SDG08", image: "/Goal-08.png" },
    { id: "SDG09", image: "/Goal-09.png" },
    { id: "SDG10", image: "/Goal-10.png" },
    { id: "SDG11", image: "/Goal-11.png" },
    { id: "SDG12", image: "/Goal-12.png" },
    { id: "SDG13", image: "/Goal-13.png" },
    { id: "SDG14", image: "/Goal-14.png" },
    { id: "SDG15", image: "/Goal-15.png" },
    { id: "SDG16", image: "/Goal-16.png" },
    { id: "SDG17", image: "/Goal-17.png" },
  ];

  // Determine which goals to display based on sdgCount prop
  const displayedGoals = sdgCount ? sdgGoals.slice(0, sdgCount) : sdgGoals;

  return (
      <>
        <Box>
          {/* Conditionally render heading and intro text if sdgCount is not provided */}
          {!sdgCount && (
              <div>
                <Typography variant="h4" align="center" gutterBottom>
                  Explore the Sustainable Development Goals
                </Typography>

                <Typography variant="h6" align="center" gutterBottom>
                  Start by clicking any of the SDGs below
                </Typography>
                <br />
                <br />
                <br /> {/* Add some spacing */}
              </div>
          )}

          {/* Grid container to display SDG goals */}
          <Grid container spacing={1} justifyContent="center" rowSpacing={4}>
            {displayedGoals.map((goal) => (
                <Grid
                    item
                    xs={12} sm={6} md={4} lg={2.5}
                    key={goal.id}
                    sx={{ padding: '0 6px' }} // Add padding to Grid items
                >
                  <Box
                      sx={{
                        width: 200,
                        height: 200,
                        border: `1px solid #a853ba`,
                        '&:hover': {
                          boxShadow: '2px 2px 2px 2px rgba(255, 255, 255, 1)' // Add a hover effect
                        },
                        alignItems: 'center'
                      }}
                  >
                    <Button
                        onClick={() => onButtonClick(goal.id)}
                        sx={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          height: "100%",
                          width: "100%",
                          mt: '0.25em' // Add margin to the top
                        }}
                    >
                      <Image
                          src={goal.image}
                          alt={goal.id}
                          width={190}
                          height={190}
                          priority={true} // Prioritize loading this image
                      />
                    </Button>
                  </Box>
                </Grid>
            ))}
          </Grid>

          {/* Conditionally render "More" button if sdgCount is provided */}
          {sdgCount && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                <Button variant="contained" onClick={() => router.push('/explore')}>
                  More
                </Button>
              </Box>
          )}
        </Box>
      </>
  );
};

export default ExploreMain;