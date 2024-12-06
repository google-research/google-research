import React from 'react';
import { useRouter }  from 'next/router';
import ExploreFlow from '@/components/Explore/ExploreFlow';
import Button from '@mui/material/Button';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import Box from '@mui/material/Box';

const Graph = () => {
  const router = useRouter();
  const sdgId = router.query.sdgId as string;

  const handleGoBack = () => {
    router.push('/explore');
  };

  return (
      <Box sx={{width: '100%', height: '80vh'}}>
        <ExploreFlow sdg_id={sdgId}/>
        <Button variant="outlined" startIcon={<ArrowBackIcon />} onClick={handleGoBack}>
          Restart
        </Button>
      </Box>
  )
}

export default Graph;