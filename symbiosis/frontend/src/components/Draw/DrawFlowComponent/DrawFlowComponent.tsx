"use client";
import React, { useCallback, useState, useRef, useEffect } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  Edge,
  MarkerType,
  ConnectionMode,
  type FitViewOptions,
  NodeChange,
  EdgeChange,
  useReactFlow,
} from '@xyflow/react';
import cytoscape from 'cytoscape';
import avsdf from 'cytoscape-avsdf';
import { useTheme } from '@mui/material/styles';
import { CircularProgress, Box, useMediaQuery } from '@mui/material';

import Sidebar from '@/components/Draw/Sidebar';
import VariableNode from '@/components/Draw/VariableNode';
import StockNode from '@/components/Draw/StockNode';
import FlowNode from '@/components/Draw/FlowNode';
import GroupNode from '@/components/Draw/GroupNode';
import CustomCircularEdge from '@/components/Draw/CustomCircularEdge/CustomCircularEdge';
import { sortNodes, getId, getNodePositionInsideParent } from '../node_utils';
import SelectedNodesToolbar from '@/components/Draw/SelectedNodesToolbar';
import CopilotSidebar from '@/components/Draw/CopilotSidebar';

cytoscape.use(avsdf); // Register the avsdf extension with Cytoscape

const proOptions = { hideAttribution: true };

// Define the custom node types used in the ReactFlow component
const nodeTypes = {
  variablenode: VariableNode,
  stocknode: StockNode,
  flownode: FlowNode,
  group: GroupNode,
};

// Define the custom edge types used in the ReactFlow component
const edgeTypes = {
  customcircular: CustomCircularEdge,
}

// Default edge options for the ReactFlow component
const defaultEdgeOptions = {
  style: {
    strokeWidth: 2,
    stroke: '#FF0072',
  },
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: '#FF0072',
  },
  animated: true,
};

// Options for fitting the view in the ReactFlow component
const fitViewOpts: FitViewOptions = {
  // maxZoom: 1,
};

// Interface for the props of the DrawFlow component
interface DrawFlowProps {
  strength: number; // Strength of the force layout
  distance: number; // Distance between nodes in the force layout
  nodes: Node[]; // Array of nodes
  edges: Edge[]; // Array of edges
  setNodes: (nodes: Node[]) => void; // Function to update the nodes state
  setEdges: (edges: Edge[]) => void; // Function to update the edges state
  onNodesChange: (changes: NodeChange[]) => void; // Callback function for node changes
  onEdgesChange: (changes: EdgeChange[]) => void; // Callback function for edge changes
  isLayoutReady: boolean; // Indicates if the layout is ready
  setIsLayoutReady: (loading: boolean) => void; // Function to update the isLayoutReady state
}

// DrawFlow component for rendering the ReactFlow instance
function DrawFlow({
                    strength = -2200, // Default strength for the force layout
                    distance = 400, // Default distance for the force layout
                    nodes,
                    edges,
                    setNodes,
                    setEdges,
                    onNodesChange,
                    onEdgesChange,
                    isLayoutReady,
                    setIsLayoutReady,
                  }: DrawFlowProps) {
  const mobileCheck = useMediaQuery('(min-width: 600px)'); // Check if the screen is mobile
  const theme = useTheme(); // Access the theme to get dynamic colors

  // Callback function to apply the avsdf layout using Cytoscape
  const applyAvsdfLayout = useCallback(() => {
    // Create a Cytoscape instance with the nodes and edges
    const cy = cytoscape({
      elements: {
        nodes: nodes.map((node) => ({
          data: { id: node.id },
        })),
        edges: edges.map((edge) => ({
          data: { source: edge.source, target: edge.target },
        })),
      },
      headless: true, // Run Cytoscape in headless mode
    });

    // Apply the avsdf layout
    const layout = cy.layout({
      name: 'avsdf',
      animate: false,
      nodeSeparation: 250,
      onLayoutStop: () => {
        setIsLayoutReady(true); // Set layout ready to true after layout is complete
      },
    });
    layout.run();

    // Update the positions of the nodes in the ReactFlow state based on the Cytoscape layout
    const updatedNodes = nodes.map((node) => {
      const cyNode = cy.getElementById(node.id);
      return {
        ...node,
        position: {
          x: cyNode.position('x'),
          y: cyNode.position('y'),
        },
      };
    });
    setNodes(updatedNodes);
  }, []);

  const initialLayoutApplied = useRef(false); // Ref to track if the initial layout has been applied

  // Apply the avsdf layout when the component mounts and the layout is not ready
  useEffect(() => {
    if (!isLayoutReady && nodes.length > 0 && edges.length > 0) {
      applyAvsdfLayout();
      initialLayoutApplied.current = true;
      setIsLayoutReady(true);
    }
  }, [isLayoutReady]);

  // Callback function to handle edge connections
  const onConnect = useCallback(
      (edge: Edge) =>
          setEdges((eds) =>
              addEdge(
                  {
                    ...edge,
                    type: 'customcircular', // Set the edge type to customcircular
                    style: {
                      strokeWidth: 2,
                      stroke: '#FF0072',
                    },
                    markerEnd: {
                      type: MarkerType.ArrowClosed,
                      color: '#FF0072',
                    },
                    animated: true,
                    data: {
                      endLabel: '',
                    }
                  },
                  eds,
              )
          ),
      [setEdges]
  );

  // Get ReactFlow utilities
  const { screenToFlowPosition, getIntersectingNodes, getNodes } = useReactFlow();

  // Handle node drop events
  const onDrop = (event: DragEvent) => {
    event.preventDefault();

    const type = event.dataTransfer.getData('application/reactflow'); // Get the node type from the drag data
    const position = screenToFlowPosition({ // Get the position in the ReactFlow coordinate system
      x: event.clientX - 20,
      y: event.clientY - 20,
    });
    const nodeDimensions = type === 'group' ? { width: 400, height: 200 } : {}; // Set dimensions for group nodes

    // Check if the dropped node intersects with any group nodes
    const intersections = getIntersectingNodes({
      x: position.x,
      y: position.y,
      width: 40,
      height: 40,
    }).filter((n) => n.type === 'group');
    const groupNode = intersections[0];

    // Create a new node object
    const newNode: Node = {
      id: getId(),
      type,
      position,
      data: { label: `${type}` },
      ...nodeDimensions,
    };

    // If the new node intersects with a group node, set its position and parent ID
    if (groupNode) {
      newNode.position = getNodePositionInsideParent(
          {
            position,
            width: 40,
            height: 40,
          },
          groupNode
      ) ?? { x: 0, y: 0 };
      newNode.parentId = groupNode?.id;
      newNode.expandParent = true;
    }

    const sortedNodes = getNodes().concat(newNode).sort(sortNodes); // Add the new node and sort the nodes
    setNodes(sortedNodes);
  };

  // Callback function when node dragging stops
  const onNodeDragStop = useCallback(
      (_: MouseEvent, node: Node) => {
        // Check if the dragged node is a variable, stock, or flow node, or if it has a parent
        if ((node.type !== 'variablenode' || node.type !== 'stocknode' || node.type !== 'flownode') && !node.parentId) {
          return;
        }

        // Check if the dragged node intersects with any group nodes
        const intersections = getIntersectingNodes(node).filter(
            (n) => n.type === 'group'
        );
        const groupNode = intersections[0];

        // If the node intersects with a group node and it's not already a child of that group, update the nodes state
        if (intersections.length && node.parentId !== groupNode?.id) {
          const nextNodes: Node[] = getNodes()
              .map((n) => {
                if (n.id === groupNode.id) {
                  return {
                    ...n,
                    className: '',
                  };
                } else if (n.id === node.id) {
                  // Calculate the position of the node inside the parent group
                  const position = getNodePositionInsideParent(n, groupNode) ?? {
                    x: 0,
                    y: 0,
                  };

                  return {
                    ...n,
                    position,
                    parentId: groupNode.id,
                    extent: 'parent',
                  } as Node;
                }

                return n;
              })
              .sort(sortNodes);

          setNodes(nextNodes);
        }
      },
      [getIntersectingNodes, getNodes, setNodes]
  );

  // Callback function while a node is being dragged
  const onNodeDrag = useCallback(
      (_: MouseEvent, node: Node) => {
        // Check if the dragged node is a variable, stock, or flow node, or if it has a parent
        if ((node.type !== 'variablenode' || node.type !== 'stocknode' || node.type !== 'flownode') && !node.parentId) {
          return;
        }

        // Check if the dragged node intersects with any group nodes
        const intersections = getIntersectingNodes(node).filter(
            (n) => n.type === 'group'
        );

        // Set the className of the group node to 'active' if the dragged node intersects with it
        const groupClassName =
            intersections.length && node.parentId !== intersections[0]?.id
                ? 'active'
                : '';

        setNodes((nds) => {
          return nds.map((n) => {
            if (n.type === 'group') {
              return {
                ...n,
                className: groupClassName,
              };
            } else if (n.id === node.id) {
              return {
                ...n,
                position: node.position,
              };
            }

            return { ...n };
          });
        });
      },
      [getIntersectingNodes, setNodes]
  );

  return (
      <>
        {/* Render the ReactFlow component */}
        <div className='drawsidebarwrapper'>
          <Sidebar /> {/* Render the Sidebar component */}
          <div className="rfWrapper">
            <ReactFlow
                className={theme.palette.background.default} // Set background color using theme
                nodes={nodes}
                edges={edges}
                onEdgesChange={onEdgesChange}
                onNodesChange={onNodesChange}
                onConnect={onConnect}
                onNodeDrag={onNodeDrag}
                onNodeDragStop={onNodeDragStop}
                onDrop={onDrop}
                onDragOver={onDragOver}
                proOptions={proOptions}
                fitView
                selectNodesOnDrag={false}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                defaultEdgeOptions={defaultEdgeOptions}
                elevateEdgesOnSelect={true}
                edgesReconnectable={true}
                connectionMode={ConnectionMode.Loose}
                defaultViewport={{
                  x: 0,
                  y: 0,
                  zoom: 1,
                }}
                onlyRenderVisibleElements={false}
            >
              <Background color="#bbb" gap={50}
                          variant={BackgroundVariant.Dots}/>{' '}
              <SelectedNodesToolbar/> {/* Render SelectedNodesToolbar */}
              <Controls/> {/* Render Controls for ReactFlow */}
              <MiniMap/> {/* Render MiniMap for ReactFlow */}
            </ReactFlow>
          </div>
        </div>
      </>
  );
}

// Interface for the props of the DrawFlowComponent component
interface DrawFlowComponentProps {
  prompt?: string; // Prompt for generating the CLD
  uuid?: any; // UUID of the paper
  modelKey?: any; // Model key for the paper
}

// DrawFlowComponent is the main component for rendering the flow
const DrawFlowComponent: React.FC<DrawFlowComponentProps> = ({
                                                               prompt,
                                                               uuid,
                                                               modelKey
                                                             }) => {
  const [defaultPrompt, setDefaultPrompt] = useState(null); // State to store the default prompt
  const [paperUuid, setPaperUuid] = useState(null); // State to store the paper UUID
  const [paperModelKey, setPaperModelKey] = useState(null); // State to store the paper model key
  const [model, setModel] = useState(null); // State to store the model data
  const ref = useRef<CopilotSidebar>(null); // Ref to access the CopilotSidebar component
  const [nodes, setNodes, onNodesChange] = useNodesState([]); // State for nodes
  const [edges, setEdges, onEdgesChange] = useEdgesState([]); // State for edges
  const [isLayoutReady, setIsLayoutReady] = useState(false); // State to track if the layout is ready

  // Use effect hook to set initial values for state variables
  useEffect(() => {
    if (prompt != null) {
      setDefaultPrompt(prompt);
    }
    if (uuid != null) {
      setPaperUuid(uuid);
    }
    if (modelKey != null) {
      setPaperModelKey(modelKey);
    }
  }, [prompt, uuid, modelKey]);

  // Use effect hook to fetch data based on paperUuid and paperModelKey
  useEffect(() => {
    const fetchData = async () => {
      try {
        const encodedModelKey = encodeURIComponent(paperModelKey);
        const encodedPath = encodeURIComponent(`/paper_graph_data?uuid=${paperUuid}&model_key=${encodedModelKey}`);
        const response = await fetch(`$$BACKEND_URL$$?path=${encodedPath}`);

        if (response.ok) {
          const data = await response.json();
          setModel(data.message[0].model);
        } else {
          console.error('No data found for the given SDG ID and paper title.');
        }
      } catch (error) {
        console.error('Error fetching or parsing model data:', error);
      }
    }
    fetchData();
  }, [paperUuid, paperModelKey]);

  // Use effect hook to execute a task when model data is available
  useEffect(() => {
    const executeTask = async () => {
      if (model) {
        // Filter out alias nodes
        const excludeNodeIds = model.nodes
            .filter((node) => ['alias'].includes(node.type))
            .map((node) => node.id);

        // Create nodes array for ReactFlow
        const nodes = model.nodes
            .filter((node) => !excludeNodeIds.includes(node.id))
            .map((node, index) => {
              const hasPosition = node.data.position && (node.data.position.x !== 0 || node.data.position.y !== 0);
              const origType = node.type;

              return {
                type: origType === 'stock' ? 'stocknode': origType === 'flow' ? 'flownode': 'variablenode',
                data: {
                  label: node.data.label?.toString(),
                },
                position: hasPosition
                    ? { x: node.data.position.x, y: node.data.position.y }
                    : { x: Math.random() * 400, y: Math.random() * 400 },
                id: node.id.toString(),
              };
            });

        // Create edges array for ReactFlow
        const edges = model.edges
            .filter((edge) => {
              const sourceNodeExists = nodes.some((node) => node.id === edge.source);
              const targetNodeExists = nodes.some((node) => node.id === edge.target);

              return sourceNodeExists && targetNodeExists;
            })
            .map((edge) => ({
              type: 'customcircular',
              id: edge.id.toString(),
              source: edge.source,
              target: edge.target,
              style: {
                strokeWidth: 2,
                stroke: '#FF0072',
              },
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: '#FF0072',
              },
              animated: true,
              data: {
                endLabel: '',
              }
            }));

        // Validate edges to ensure they have valid source and target nodes
        edges.forEach((edge) => {
          const sourceNodeExists = nodes.some((node) => node.id === edge.source);
          const targetNodeExists = nodes.some((node) => node.id === edge.target); 1

          if (!sourceNodeExists || !targetNodeExists) {
            console.error(`Edge with id "${edge.id}" has an invalid source or target:`, edge);
          }
        });

        setNodes(nodes); // Update the nodes state

        // Set a timeout to update the edges state and set the layout as ready
        setTimeout(() => {
          setEdges(edges);
          setIsLayoutReady(true);
        }, 5000);
      }
    };
    executeTask();
  }, [model]);

  return (
      <ReactFlowProvider>
        {/* Render the DrawFlow component */}
        <DrawFlow
            strength={-1000}
            distance={500}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodes={nodes}
            edges={edges}
            setNodes={setNodes}
            setEdges={setEdges}
            isLayoutReady={isLayoutReady}
            setIsLayoutReady={setIsLayoutReady}
        />
        {/* Render the CopilotSidebar component */}
        <CopilotSidebar
            prompt={defaultPrompt}
            setNodes={setNodes}
            setEdges={setEdges}
            nodes={nodes}
            edges={edges}
            isLayoutReady={isLayoutReady}
            setIsLayoutReady={setIsLayoutReady}
        />
      </ReactFlowProvider>
  );
};

export default DrawFlowComponent;