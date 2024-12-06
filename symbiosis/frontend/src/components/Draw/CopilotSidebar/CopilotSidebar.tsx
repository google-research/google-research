// CopilotSidebar.tsx
"use client";
import React, { useEffect, useState, useRef } from 'react';
import { useChat } from 'ai/react';
import {
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemText,
  TextField,
  Typography,
  Box,
  Card,
  CardContent,
  CircularProgress,
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import SendIcon from '@mui/icons-material/Send';
import { useTheme } from '@mui/material/styles';
import { keyframes } from '@emotion/react';
import Tooltip from '@mui/material/Tooltip';
import ReactMarkdown from 'react-markdown';
import { useSession } from 'next-auth/react';
import Avatar from '@mui/material/Avatar';
import Image from 'next/image';
import StopCircleIcon from '@mui/icons-material/StopCircle';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import { Node, Edge, MarkerType } from '@xyflow/react';

// Keyframes for the bounce animation of the chat icon
const bounce = keyframes`
  0%, 20%, 50%, 80%, 100% {
    transform: translateY(0);
  }
  40% {
    transform: translateY(-10px);
  }
  60% {
    transform: translateY(-5px);
  }
`;

// Keyframes for the rotate animation of the AI avatar
const rotate = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

// Interface for the props of the CopilotSidebar component
interface CopilotSidebarProps {
  prompt: string | null; // Prompt for generating the CLD (can be null)
  setNodes: (nodes: Node[]) => void; // Function to update the nodes in the diagram
  setEdges: (edges: Edge[]) => void; // Function to update the edges in the diagram
  nodes: Node[]; // Current nodes in the diagram
  edges: Edge[]; // Current edges in the diagram
  isLayoutReady: boolean; // Indicates if the layout of the diagram is ready
  setIsLayoutReady: (loading: boolean) => void; // Function to update the isLayoutReady state
}

const CopilotSidebar = ({
                          prompt,
                          setNodes,
                          setEdges,
                          nodes,
                          edges,
                          isLayoutReady,
                          setIsLayoutReady
                        }: CopilotSidebarProps) => {
  const [suggestions, setSuggestions] = useState<string[]>([]); // State to store suggestions
  const [cardTexts, setCardTexts] = useState<string[]>([]); // State to store the texts displayed on the suggestion cards
  const firstRender = useRef(true); // Ref to track the first render of the component

  // Construct the initial message to be sent to the chat API
  const initialMessage = `Use this causal loop diagram data in context of the chat.
  Do not respond if the user query is anything else other than query related to the causal loop diagram.
  Variables: ${JSON.stringify(nodes)}
  Causal Links: ${JSON.stringify(edges)}
  edgeLabel in Causal Links are polarity of the causal links.`;

  // Construct the prompt for generating suggestions
  const suggestionPrompt = `Use this causal loop diagram data in context and provide suggestions. 
  Variables: ${JSON.stringify(nodes)}
  Causal Links: ${JSON.stringify(edges)}`;

  // Initialize the chat using the useChat hook
  const { messages, input, handleInputChange, handleSubmit, append, isLoading, stop, error, reload } = useChat({
    api: '/api/stream-text', // API endpoint for streaming text
    keepLastMessageOnError: false,
    sendExtraMessageFields: true,
    initialMessages: [
      {
        id: '',
        role: 'system',
        data: '',
        content: "👋 Hello! I'm your friendly **systems thinking co-pilot**. 🚀 Ask me anything! Or choose one of the **suggested prompts** below to get started. 👇",
      },
      {
        id: '',
        role: 'system',
        data: '',
        content: initialMessage, // Send the initial message with CLD data
      },
    ],
  });

  // Fetch suggestions from the API
  const getSuggestions = async () => {
    const response = await fetch('/api/get-suggestions', {
      method: 'POST',
      body: JSON.stringify({
        prompt: suggestionPrompt, // Send the prompt for generating suggestions
      }),
    });
    const json = await response.json();
    setSuggestions(json.suggestions); // Update the suggestions state
  };

  // Fetch CLD data from the API
  const getCLD = async () => {
    if (!prompt) return; // Don't fetch if the prompt is null or undefined

    try {
      const response = await fetch('/api/generate-cld', {
        method: 'POST',
        body: JSON.stringify({
          prompt: prompt, // Send the prompt for generating the CLD
        }),
      });

      const json = await response.json();

      // Transform the causal links to include markerEnd information for the diagram
      const transformedCausalLinks = json.causallinks.map((edge) => ({
        ...edge,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edge.markerEnd.color,
        },
      }));

      setNodes(json.variables); // Update the nodes state
      setEdges(transformedCausalLinks); // Update the edges state
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  // useEffect to fetch CLD data when the prompt changes
  useEffect(() => {
    getCLD();
    setIsLayoutReady(true);
  }, [prompt]);

  // useEffect to fetch suggestions when nodes or edges change, but not on the first render
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    if (isLayoutReady) {
      getSuggestions();
    }
  }, [nodes, edges, isLayoutReady]);

  // useEffect to update the card texts when suggestions change
  useEffect(() => {
    setCardTexts(suggestions.slice(0, 4));
  }, [suggestions]);

  // State to control the visibility of the drawer and access the theme
  const [open, setOpen] = React.useState(false);
  const theme = useTheme();
  const { data: session } = useSession(); // Get the user session
  const userProfileImg = session?.user?.image as string; // Get the user profile image

  // Function to toggle the drawer
  const handleToggle = () => {
    setOpen(!open);
  };

  // Log errors if any
  if (error) {
    console.log(error.stack);
  }

  return (
      <div>
        {/* Button to open the Co-pilot sidebar */}
        <Tooltip title="Chat with Co-pilot" arrow>
          <IconButton
              sx={{
                position: 'fixed',
                top: 100,
                right: 16,
                zIndex: 1000,
                backgroundColor: theme.palette.secondary.main,
                color: theme.palette.primary.main,
                borderRadius: '50%',
                padding: 2,
                '&:hover': {
                  animation: 'none',
                  backgroundColor: theme.palette.primary.main,
                  color: theme.palette.secondary.main,
                },
                animation: `${bounce} 1s ease infinite`, // Apply bounce animation
              }}
              onClick={handleToggle}
              color="primary"
          >
            <ChatIcon />
          </IconButton>
        </Tooltip>

        {/* Drawer for the Co-pilot sidebar */}
        <Drawer
            anchor="right"
            open={open}
            onClose={handleToggle}
            PaperProps={{
              sx: {
                top: 64,
                width: { xs: '100%', sm: 600 }, // Responsive width
                backgroundColor: theme.palette.primary,
                maxHeight: '95vh',
              },
            }}
        >
          {/* Content of the Co-pilot sidebar */}
          <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
              }}
          >
            {/* Title of the Co-pilot sidebar */}
            <Typography
                variant="h6"
                sx={{
                  fontFamily: 'Roboto Condensed',
                  fontWeight: 700,
                  color: theme.palette.primary.contrastText,
                  padding: 2,
                  borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
                  textAlign: 'center',
                }}
                gutterBottom
            >
              Co-pilot
            </Typography>

            {/* List of chat messages */}
            <List
                sx={{
                  p: 2,
                  display: 'flex',
                  flexDirection: 'column',
                  height: '100%',
                  overflowY: 'auto',
                }}
            >
              <Box sx={{ flexGrow: 1 }}>
                {/* Display chat messages */}
                {messages
                    .filter((m) => m.content !== initialMessage) // Filter out the initial message
                    .map((m) => (
                        <ListItem
                            key={m.id}
                            sx={{
                              display: 'flex',
                              alignItems: 'flex-start',
                              justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
                              gap: '10px',
                            }}
                        >
                          {/* Chat message bubble */}
                          <Box
                              sx={{
                                backgroundColor: m.role === 'user' ? 'primary.main' : 'grey.300',
                                color: m.role === 'user' ? 'white' : 'black',
                                borderRadius: '10px',
                                padding: '10px',
                                maxWidth: '85%',
                                display: 'inline-block',
                              }}
                          >
                            {/* Render message content as Markdown */}
                            <ReactMarkdown
                                components={{
                                  p: ({ node, ...props }) => <p style={{ fontSize: '0.9rem' }} {...props} />,
                                  h1: ({ node, ...props }) => <h1 style={{ fontSize: '1.4rem' }} {...props} />,
                                  h2: ({ node, ...props }) => <h2 style={{ fontSize: '1.3rem' }} {...props} />,
                                  h3: ({ node, ...props }) => <h3 style={{ fontSize: '1.2rem' }} {...props} />,
                                  h4: ({ node, ...props }) => <h4 style={{ fontSize: '1.1rem' }} {...props} />,
                                  ul: ({ node, ...props }) => <ul style={{ fontSize: '0.9rem' }} {...props} />,
                                }}
                            >
                              {m.content}
                            </ReactMarkdown>
                          </Box>

                          {/* Display user or AI avatar */}
                          {m.role === 'user' ? (
                              <IconButton sx={{ p: 0 }}>
                                <Avatar alt={session?.user?.name as string} src={userProfileImg} />
                              </IconButton>
                          ) : (
                              <IconButton sx={{ p: 0 }}>
                                <div>
                                  <Avatar
                                      sx={{
                                        m: 1,
                                        bgcolor: 'transparent',
                                        animation: `${rotate} 4s linear 2`, // Apply rotate animation
                                      }}
                                  >
                                    <Image src="/Gemini_Logo.png" alt="AI" width={24} height={24} />
                                  </Avatar>
                                </div>
                              </IconButton>
                          )}
                        </ListItem>
                    ))}

                {/* Loading indicator */}
                {isLoading && (
                    <div>
                      <CircularProgress />
                      <StopCircleIcon onClick={() => stop()} /> {/* Stop button */}
                    </div>
                )}

                {/* Error message and reload button */}
                {error && (
                    <>
                      <div>An error occurred while generating a response from the co-pilot.</div>
                      <AutorenewIcon onClick={() => reload()} /> {/* Reload button */}
                    </>
                )}
              </Box>

              {/* Suggested prompts */}
              <Typography variant="body2" color="text.secondary">
                Suggested prompts:
              </Typography>
              <Box
                  sx={{
                    mt: 2,
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)', // Two columns
                    gap: 2,
                  }}
              >
                {/* Display suggestion cards */}
                {cardTexts &&
                    cardTexts.slice(0, 4).map((cardText, index) => (
                        <Card
                            variant="outlined"
                            key={index}
                            sx={{
                              cursor: 'pointer',
                              borderRadius: 2,
                              p: 1,
                              maxHeight: 60,
                              maxWidth: 400,
                              '&:hover': {
                                boxShadow: '0px 2px 5px 0px white',
                              },
                            }}
                            // Append the clicked suggestion to the chat and update the card texts
                            onClick={() => {
                              append({
                                role: 'user',
                                content: cardText,
                              });

                              setCardTexts((prevCardTexts) => {
                                const updatedCardTexts = prevCardTexts.filter((ct) => ct !== cardText);
                                if (suggestions.length > updatedCardTexts.length) {
                                  updatedCardTexts.push(suggestions[updatedCardTexts.length]);
                                }
                                return updatedCardTexts;
                              });
                            }}
                        >
                          <CardContent
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                          >
                            <Typography
                                variant="body2"
                                color="text.secondary"
                                sx={{
                                  fontSize: '0.7rem',
                                  wordWrap: 'break-word',
                                }}
                            >
                              {cardText}
                            </Typography>
                          </CardContent>
                        </Card>
                    ))}
              </Box>

              {/* Input field for the user message */}
              <br />
              <Box component="form" onSubmit={handleSubmit} sx={{ mt: 'auto', display: 'flex' }}>
                <TextField
                    fullWidth
                    placeholder="Ask anything..."
                    value={input}
                    onChange={handleInputChange}
                    sx={{ mb: 1, input: { color: 'black' }, flexGrow: 1 }}
                />
                {/* Send button */}
                <IconButton type="submit" color="primary">
                  <SendIcon />
                </IconButton>
              </Box>
            </List>
          </div>
        </Drawer>
      </div>
  );
};

export default CopilotSidebar;