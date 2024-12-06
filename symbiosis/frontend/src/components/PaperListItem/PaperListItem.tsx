import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  Tooltip,
  Card,
  CardContent,
  Typography,
  Collapse,
  Divider,
  Box,
  Chip
} from '@mui/material';

// Interface for the props of the PaperListItem component
interface PaperListItemProps {
  paper: {
    uuid: string; // Unique identifier for the paper
    paper_title: string; // Title of the paper
    abstract: string; // Abstract of the paper
    sdg_goals: string[]; // Array of SDG goals related to the paper
    sdg_targets: string[]; // Array of SDG targets related to the paper
    models: string; // Stringified JSON object representing the models used in the paper
  };
  selectedNodeId: string; // ID of the selected node in the diagram (not used in this component)
}

// PaperListItem component for displaying information about a paper
function PaperListItem({ paper, selectedNodeId }: PaperListItemProps) {
  const [open, setOpen] = useState(false); // State to control the expansion of the paper details
  const router = useRouter(); // Access the router for navigation

  useEffect(() => {
  }, [paper]);

  // Truncate the abstract if it's longer than 100 characters
  const truncatedAbstract = paper.abstract.length > 100
      ? paper.abstract.substring(0, 100) + "..."
      : paper.abstract;

  // Toggle the expansion state of the paper details
  const handleClick = () => {
    setOpen(!open);
  };

  // Navigate to the datatable page with the paper's UUID as a query parameter
  const handleCardClick = () => {
    router.push(`/datatable?uuid=${paper.uuid}`);
  };

  return (
      <Tooltip title={paper.abstract} placement="left"> {/* Tooltip to display the full abstract on hover */}
        <Card
            variant="outlined"
            sx={{
              mb: 2,
              boxShadow: 3,
              padding: 0.1,
              margin: 1.5,
              borderRadius: 1,
              backgroundColor: '#6d45b5' // Set the background color
            }}
            onClick={handleCardClick} // Navigate to the datatable page when the card is clicked
        >
          <CardContent>
            {/* Paper title */}
            <Typography
                variant="subtitle2"
                component="div"
                onClick={handleClick} // Expand/collapse the details when the title is clicked
                sx={{
                  cursor: 'pointer',
                  textWrap: 'wrap',
                  fontFamily: 'Monospace',
                  fontWeight: 500,
                  fontSize: 12,
                }}
            >
              {paper.paper_title}
            </Typography>

            {/* Collapsible details */}
            <Collapse in={open} timeout="auto" unmountOnExit>
              <Divider /> {/* Divider to separate the title from the details */}
              <Typography variant="body1" sx={{ mt: 1, fontWeight: 'bold' }}>Abstract:</Typography>
              {/* Truncated abstract */}
              <Typography
                  variant="body2"
                  sx={{
                    mt: 1,
                    whiteSpace: 'normal',
                    overflowWrap: 'break-word'
                  }}
              >
                {truncatedAbstract}
              </Typography>

              <Divider /> {/* Divider to separate the abstract from the SDG goals */}

              {/* SDG goals */}
              {paper.sdg_goals && (
                  <>
                    <Typography variant="body1" sx={{ mt: 1, fontWeight: 'bold' }}>SDG Goals:</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {paper.sdg_goals.map((goal, idx) => (
                          <Chip key={idx} label={goal} variant="outlined" size="small" />
                      ))}
                    </Box>
                    <Divider /> {/* Divider to separate the SDG goals from the SDG targets */}
                    {/* SDG targets */}
                    <Typography variant="body1" sx={{ mt: 1, fontWeight: 'bold' }}>SDG Targets:</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                      {paper.sdg_targets.map((target, idx) => (
                          <Chip key={idx} label={target} variant="outlined" size="small" />
                      ))}
                    </Box>
                  </>
              )}
            </Collapse>

            {/* Number of models */}
            {paper.models != null ? (
                <Chip
                    label={`Models: ${Object.keys(JSON.parse(paper.models)).length}`}
                    sx={{ mt: 1 }}
                />
            ) : (
                <Chip
                    label={`Models: 0`}
                    sx={{ mt: 1 }}
                />
            )}
          </CardContent>
        </Card>
      </Tooltip>
  );
}

export default PaperListItem;