import React, { type FC, useRef, useEffect, useState } from 'react';
import {
  useInternalNode,
  EdgeLabelRenderer,
  type Edge,
  type EdgeProps
} from '@xyflow/react';
import { getEdgeParams } from '../utils.js';
import { Box, TextField } from '@mui/material';

// Component to render an editable label for an edge
function EdgeLabel({ id, label }) {
  const [labelText, setLabelText] = useState(label); // State to store the label text
  const textRef = useRef<HTMLInputElement>(null); // Ref to access the input element

  // Handle changes in the input field
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setLabelText(event.target.value);
  };

  // Focus the input field when the mouse enters the label
  const handleMouseEnter = () => {
    if (textRef.current) {
      textRef.current.focus();
    }
  };

  // Focus the input field when the label is clicked
  const handleClick = () => {
    if (textRef.current) {
      textRef.current.focus();
    }
  };

  return (
      <>
        {/* Div to wrap the label */}
        <div
            id={id}
            style={{
              position: 'absolute',
              background: 'transparent',
              padding: 10,
              color: '#ff5050',
              fontSize: 18,
              fontWeight: 700,
              pointerEvents: 'all', // Allow pointer events on the label
            }}
            className="nodrag nopan"
        >
          {/* Box to wrap the TextField */}
          <Box
              component="form"
              sx={{
                '& .MuiTextField-root': {
                  width: '4ch',
                  alignItems: 'center',
                }
              }}
              noValidate
              autoComplete="off"
          >
            {/* TextField for the label */}
            <TextField
                inputRef={textRef}
                value={labelText ? labelText : "?"} // Display "?" if labelText is empty
                onChange={handleChange}
                onMouseEnter={handleMouseEnter}
                onClick={handleClick}
                size="small"
                margin="dense"
                sx={{
                  mt: 0,
                  '& .MuiOutlinedInput-root': {
                    '& fieldset': {
                      border: 'none',
                    },
                    '&:hover': {
                      boxShadow: '0px 2px 4px 0px rgba(255, 255, 255, 0.4)'
                    },
                  },
                  '& .MuiInputBase-input': {
                    textAlign: 'center',
                    fontSize: labelText === '-' ? '32px' : '16px', // Adjust font size based on labelText
                  },
                }}
            />
          </Box>
        </div>
      </>
  );
}

// Component to render a custom circular edge with an editable label
const CustomCircularEdge: FC<EdgeProps<Edge<{ endLabel: string }>>> = ({
                                                                         id, // Unique ID for the edge
                                                                         source, // ID of the source node
                                                                         target, // ID of the target node
                                                                         markerEnd, // Marker to be displayed at the end of the edge
                                                                         style, // Custom style object for the edge
                                                                         data, // Data associated with the edge
                                                                       }) => {
  const sourceNode = useInternalNode(source); // Get the source node
  const targetNode = useInternalNode(target); // Get the target node
  const pathRef = useRef<SVGPathElement>(null); // Ref to access the path element
  const labelId = `${id}-label`; // Unique ID for the label

  // Return null if either source or target node is not found
  if (!sourceNode || !targetNode) {
    return null;
  }

  // Get the parameters for the edge path
  const { sx, sy, tx, ty, sourcePos, targetPos } = getEdgeParams(
      sourceNode,
      targetNode
  );

  // Calculate the radius for the circular arc
  // Fallback to 100 if the calculation results in NaN
  const radiusX = Number.isNaN(tx - sx) ? 100 : Math.abs(tx - sx) / 2 + 280;
  const radiusY = Number.isNaN(ty - sy) ? 100 : Math.abs(ty - sy) / 2 + 280;

  // Construct the SVG path for the circular arc
  const sweepFlag = 0;
  const path = `M ${sx},${sy} A ${radiusX} ${radiusY} 0 0 ${sweepFlag} ${tx},${ty}`;

  // Calculate the midpoint of the edge
  const midpointX = (sx + tx) / 2;
  const midpointY = (sy + ty) / 2;

  // Log an error if any NaN values are encountered in the calculations
  if (Number.isNaN(radiusX) || Number.isNaN(radiusY) || Number.isNaN(sx) || Number.isNaN(sy) || Number.isNaN(tx) || Number.isNaN(ty)) {
    console.error(
        'NaN values encountered in edge calculations!',
        { id, source, target, sourceNode, targetNode, radiusX, radiusY, sx, sy, tx, ty }
    );
  }

  return (
      <>
        {/* Render the edge as an SVG path */}
        <path
            ref={pathRef}
            id={id}
            style={style}
            d={path}
            markerEnd={markerEnd}
            fill="none"
        />

        {/* Render the label using EdgeLabelRenderer */}
        <EdgeLabelRenderer>
          <div
              id={labelId}
              style={{
                position: 'absolute',
                transform: `translate(-50%, -50%) translate(${midpointX}px,${midpointY}px)`,
                fontSize: 12,
                pointerEvents: 'all', // Allow pointer events on the label
              }}
              className="nodrag nopan"
          >
            <EdgeLabel id={labelId} label={data.endLabel} /> {/* Render the editable label */}
          </div>
        </EdgeLabelRenderer>
      </>
  );
};

export default CustomCircularEdge;