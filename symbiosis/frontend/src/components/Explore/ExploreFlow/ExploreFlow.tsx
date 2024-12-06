import React, { useCallback, useState, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  useNodesState,
  useEdgesState,
  addEdge,
  Node,
  Edge,
  OnNodesChange,
  OnEdgesChange,
} from '@xyflow/react';
import FocusNode from '@/components/Explore/FocusNode';
import { FocusNodeData } from '@/components/Explore/FocusNode';
import CustomEdge from '@/components/Explore/CustomEdge';
import GraphNode from '@/components/Explore/GraphNode';
import { GraphNodeData } from '@/components/Explore/GraphNode';
import PaperListItem from '@/components/PaperListItem';

import '@xyflow/react/dist/base.css';

import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import { GoGoal } from "react-icons/go";
import SportsScoreIcon from '@mui/icons-material/SportsScore';
import dagre from 'dagre';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Tooltip from '@mui/material/Tooltip';
import { useMediaQuery } from '@mui/material';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';

const proOptions = { hideAttribution: true };

// Define the custom node types used in the ReactFlow component
const nodeTypes = {
  focus: FocusNode,
  graph: GraphNode,
};

// Define the custom edge types used in the ReactFlow component
const edgeTypes = {
  edge: CustomEdge,
};

// Default options for edges in the ReactFlow component
const defaultEdgeOptions = {
  type: 'edge',
  markerEnd: 'edge-circle',
}

// Define the interface for the props of the ExploreFlow component
interface ExploreFlowProps {
  sdg_id: string; // The ID of the SDG to display
}

// Define the interface for the data of a node
interface NodeData {
  icon: React.ReactNode;
  title: string;
  subline: string;
  id: string;
  type: string;
  sdgLabel: string;
  sdgDesc: string;
  sdgId: string;
  width: any;
  height: any;
}

// Define the interface for the data of an edge
interface EdgeData {
  id: string;
  source: string;
  target: string;
}

// Component to provide data and functionality to the ReactFlow component
function ExploreFlowProvider({
                               sdg_id = "SDG1", // Default SDG ID
                               nodes, // Array of nodes
                               edges, // Array of edges
                               onNodesChange, // Callback function for node changes
                               onEdgesChange, // Callback function for edge changes
                               setNodes, // Function to update the nodes state
                               setEdges // Function to update the edges state
                             }: {
  sdg_id?: string;
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
}) {
  const [error, setError] = React.useState<string | null>(null);
  const [allPapers, setAllPapers] = useState({}); // State to store all papers
  const [uniquePapers, setUniquePapers] = useState({}); // State to store unique papers
  const [selectedNodeId, setSelectedNodeId] = useState(null); // State to store the ID of the selected node
  const [paperCount, setPaperCount] = useState(0); // State to store the count of papers
  const theme = useTheme();
  const colorMode = theme.palette.mode === 'dark' ? 'dark' : 'light'; // Get color mode from theme

  useEffect(() => {
    // Fetch data for the flow chart
    async function fetchData() {
      try {
        const encodedPath = encodeURIComponent('/sdg_data');
        const response = await fetch(`$$BACKEND_URL$$?path=${encodedPath}`);
        const data = await response.json();

        if (response.ok) {
          // Transform the fetched node data to the format required by ReactFlow
          const inputNodes = data.message.nodes.map((node: NodeData) => ({
            ...node,
            id: node.id,
            position: { x: 0, y: 0 }, // Initial position
            data: {
              icon: node.type === 'goal' ? <SportsScoreIcon /> : <GoGoal />, // Set icon based on node type
              title: node.sdgLabel,
              subline: node.sdgDesc,
              id: node.sdgId,
            },
            type: node.type == 'goal' ? 'focus' : 'graph', // Set node type based on data
          }));

          // Transform the fetched edge data to the format required by ReactFlow
          const inputEdges = data.message.edges.map((edge: EdgeData) => ({
            ...edge,
            id: edge.id,
            source: edge.source,
            target: edge.target,
          }));

          // Use the `dagre` library to automatically lay out the nodes in the graph
          const dagreGraph = new dagre.graphlib.Graph();
          dagreGraph.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 100 });
          dagreGraph.setDefaultEdgeLabel(() => ({}));

          // Set nodes in the dagre graph
          inputNodes.forEach((node: NodeData) => {
            dagreGraph.setNode(node.id, {
              width: node.width ?? 350,
              height: node.height ?? 36
            });
          });

          // Set edges in the dagre graph
          inputEdges.forEach((edge: EdgeData) => {
            dagreGraph.setEdge(edge.source, edge.target);
          });

          dagre.layout(dagreGraph);

          // Update the positions of the nodes based on the dagre layout
          const layoutedNodes = inputNodes.map((node: NodeData) => {
            const { x, y } = dagreGraph.node(node.id);
            const position = {
              x: x - (node.width ?? 0) / 2,
              y: y - (node.height ?? 0) / 2,
            };

            return { ...node, position };
          });

          // Update the state with the layouted nodes and edges
          setNodes(layoutedNodes);
          setEdges(inputEdges);

        } else {
          // Handle error if the response is not ok
          setError(data.error || "Error fetching data from server");
        }
      } catch (error) {
        console.error('Error fetching or parsing data:', error);
        setError('An error occurred while fetching data.');
      }
    }

    fetchData(); // Call the fetchData function
  }, [sdg_id]); // Re-fetch data when sdg_id changes

  const [drawerOpen, setDrawerOpen] = useState(false); // State to control the drawer visibility
  const [selectedNodeData, setSelectedNodeData] = useState<FocusNodeData>(null); // State to store data of the selected node

  // Handle node click event
  const handleNodeClick = async (event: React.MouseEvent, node: Node) => {
    setSelectedNodeData(node.data);
    setSelectedNodeId(node.id);
    setDrawerOpen(true); // Open the drawer

    try {
      // Fetch papers related to the selected node
      const encodedPath = encodeURIComponent(`/sdg_papers?sdg_id=${node.data.id}`);
      const response = await fetch(`$$BACKEND_URL$$?path=${encodedPath}`);

      if (response.ok) {
        const data = await response.json();
        setPaperCount(Object.keys(data.message).length);

        // Process and filter papers to get unique papers by title
        const papersByTitle = new Map();
        for (const paper of data.message) {
          if (!papersByTitle.has(paper.paper_title)) {
            papersByTitle.set(paper.paper_title, []);
          }
          papersByTitle.get(paper.paper_title).push(paper);
        }

        const uniquePapersArray = [];
        for (const [title, papers] of papersByTitle.entries()) {
          let modelCount = 0;
          const modelSets = new Set();

          for (const paper of papers) {
            if (paper.models) {
              const modelsArray = JSON.parse(paper.models);
              if (Array.isArray(modelsArray)) {
                modelsArray.forEach(model => modelSets.add(model));
              }
            }
          }

          modelCount = modelSets.size;

          const representativePaper = papers[0];
          uniquePapersArray.push({ ...representativePaper, model_count: modelCount });
        }

        // Sort unique papers by model count
        uniquePapersArray.sort((a, b) => b.model_count - a.model_count);

        // Update the uniquePapers state with the processed papers
        setUniquePapers(prevUniquePapers => ({
          ...prevUniquePapers,
          [node.data.title]: uniquePapersArray
        }));

      } else {
        console.error("Error fetching papers:", response.status);
      }
    } catch (error) {
      console.error("Error fetching papers:", error);
    }
  };

  // Callback function when a new connection is created in the ReactFlow component
  const onConnect = useCallback((params) => setEdges((els) => addEdge(params, els)), [setEdges]);

  // Check if the screen is mobile using a media query
  const mobileCheck = useMediaQuery("(min-width: 600px)");

  // Filter nodes and edges based on the selected SDG ID
  const initialFilteredNodes = nodes.filter(node => node.sdgId === sdg_id);
  const connectedNodeIds = new Set<string>();
  for (const edge of edges) {
    if (initialFilteredNodes.some(node => node.id === edge.source || node.id === edge.target)) {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    }
  }
  const allFilteredNodes = nodes.filter(node =>
      node.sdgId === sdg_id || connectedNodeIds.has(node.id)
  );
  const allFilteredEdges = edges.filter(edge =>
      allFilteredNodes.some(node => node.id === edge.source) &&
      allFilteredNodes.some(node => node.id === edge.target)
  );

  // Handle mouse enter event on a node
  const handleNodeMouseEnter = (_: React.MouseEvent, node: Node) => {
    if (node.type === 'graph') {
      setNodes((nodes) =>
          nodes.map((n) =>
              n.id === node.id ? {
                ...n,
                data: { ...n.data, toolbarEnabled: true }
              } : n
          )
      );
    }
  }

  // Handle mouse leave event on a node
  const handleNodeMouseLeave = (_: React.MouseEvent, node: Node) => {
    if (node.type === 'graph') {
      setNodes((nodes) =>
          nodes.map((n) =>
              n.id === node.id ? {
                ...n,
                data: { ...n.data, toolbarEnabled: false }
              } : n
          )
      );
    }
  }

  return (
      <>
        {/* ReactFlow component to render the flow chart */}
        <ReactFlow
            nodes={allFilteredNodes}
            edges={allFilteredEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultEdgeOptions={defaultEdgeOptions}
            proOptions={proOptions}
            onNodeClick={handleNodeClick}
            colorMode="system"
            onNodeMouseEnter={handleNodeMouseEnter}
            onNodeMouseLeave={handleNodeMouseLeave}
        >
          <Controls showInteractive={false} /> {/* Show basic controls */}

          {/* Define SVG elements for the edges */}
          <svg>
            <defs>
              <linearGradient id="edge-gradient"> {/* Gradient for the edge */}
                <stop offset="0%" stopColor="#ae53ba" />
                <stop offset="100%" stopColor="#2a8af6" />
              </linearGradient>

              {/* Marker for the end of the edge */}
              <marker
                  id="edge-circle"
                  viewBox="-5 -5 10 10"
                  refX="0"
                  refY="0"
                  markerUnits="strokeWidth"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto"
              >
                <circle stroke="#2a8af6" strokeOpacity="0.75" r="2" cx="0" cy="0" />
              </marker>
            </defs>
          </svg>
        </ReactFlow>

        {/* Drawer component to display information about the selected node */}
        <Drawer
            anchor="right"
            open={drawerOpen}
            onClose={() => setDrawerOpen(false)}
            sx={{
              [`& .MuiDrawer-paper`]: {
                right: 0,
                top: mobileCheck ? 64 : 57, // Adjust top position based on screen size
                flexShrink: 0,
                whiteSpace: "nowrap",
                boxSizing: "border-box",
                backgroundColor: 'black',
                color: 'white',
                border: '2px solid #673ab7',
                fontFamily: 'Monospace',
                fontWeight: 500,
                letterSpacing: '-0.2px',
                fontSize: 12,
                maxHeight: `calc(100vh - 68px)`,
              },
            }}
        >
          {/* Content of the drawer */}
          <Box sx={{ width: 350 }}>
            <Box>
              {/* Display the title of the selected node */}
              <Typography
                  sx={{
                    p: 1,
                    fontSize: 18,
                    textWrap: 'wrap'
                  }}
              >
                {selectedNodeData?.id}: {selectedNodeData?.title}
              </Typography>
              {/* Display the total paper count */}
              <Box sx={{ display: 'flex', alignItems: 'center', margin: 1 }}>
                <Chip label={`Total Paper Count: (${paperCount})`} sx={{ backgroundColor: '#673ab7' }} />
              </Box>
              <Divider />
            </Box>

            {/* Display the list of unique papers */}
            <Box sx={{
              // overflowY: 'auto',
              // maxHeight: 'calc(100vh - 100px)'
            }}>
              <List>
                {(uniquePapers[selectedNodeData?.title] || []).map((paper, index) => (
                    <PaperListItem
                        key={paper.uuid || index}
                        paper={paper}
                        selectedNodeId={selectedNodeId}
                    />
                ))}
              </List>
            </Box>
          </Box>
        </Drawer>
      </>
  );
}

// Main component for the Explore Flow page
const ExploreFlow: React.FC<ExploreFlowProps> = ({ sdg_id }) => {
  // Initialize nodes and edges state using useNodesState and useEdgesState hooks
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  return (
      <Box sx={{ width: '100%', height: '80vh' }}> {/* Container for the flow chart */}
        {/* Provide the nodes, edges, and event handlers to the ExploreFlowProvider */}
        <ExploreFlowProvider
            sdg_id={sdg_id}
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            setNodes={setNodes}
            setEdges={setEdges}
        />
        {/* Display a message to the user */}
        <Typography variant="h5" align="center" gutterBottom>
          Start by clicking any of the SDG nodes above
        </Typography>
      </Box>
  );
};

export default ExploreFlow;