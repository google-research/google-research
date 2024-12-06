import { memo, useState, useRef } from 'react';
import {
  Handle,
  Position,
  NodeToolbar,
  NodeProps,
  useStore,
  useReactFlow,
  type Node,
} from '@xyflow/react';
import {
  Box,
  TextField,
  IconButton,
  Tooltip
} from '@mui/material';
import DeleteOutlinedIcon from '@mui/icons-material/DeleteOutlined';
import TabUnselectedIcon from '@mui/icons-material/TabUnselected';
import SpeakerIcon from '@mui/icons-material/Speaker';

import useDetachNodes from '../detach_utils';

import { useTheme } from '@mui/material/styles';

// FlowNode component for rendering a Flow node in the diagram
function FlowNode({ id, data }: NodeProps<Node>) {
  const hasParent = useStore((store) => !!store.nodeLookup.get(id)?.parentId); // Check if the node has a parent
  const { deleteElements } = useReactFlow(); // Get the deleteElements function from useReactFlow
  const detachNodes = useDetachNodes(); // Custom hook for detaching nodes
  const [nodeName, setNodeName] = useState(data?.label || ''); // State to store the node name
  const textRef = useRef<HTMLInputElement>(null); // Ref to access the input element
  const theme = useTheme(); // Access the theme to get dynamic colors

  // Callback function to delete the node
  const onDelete = () => deleteElements({ nodes: [{ id }] });

  // Callback function to detach the node from its parent
  const onDetach = () => detachNodes([id]);

  // Callback function to handle changes in the input field
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNodeName(event.target.value);
  };

  // Focus the input field when the mouse enters the node
  const handleMouseEnter = () => {
    if (textRef.current) {
      textRef.current.focus();
    }
  };

  // Focus the input field when the node is clicked
  const handleClick = () => {
    if (textRef.current) {
      textRef.current.focus();
    }
  };

  return (
      <>
        {/* NodeToolbar for the delete and detach buttons */}
        <NodeToolbar className="nodrag">
          {/* Tooltip for the delete button */}
          <Tooltip title="Delete Node">
            <IconButton
                onClick={onDelete}
                size="small"
                sx={{
                  bgcolor: 'none',
                  '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
                }}
            >
              <DeleteOutlinedIcon sx={{ color: theme.palette.primary.main }} />
            </IconButton>
          </Tooltip>

          {/* Conditionally render the detach button if the node has a parent */}
          {hasParent && (
              <Tooltip title="Detach Node">
                <IconButton
                    onClick={onDetach}
                    size="small"
                    sx={{
                      bgcolor: 'none',
                      '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
                    }}
                >
                  <TabUnselectedIcon sx={{ color: theme.palette.primary.main }} />
                </IconButton>
              </Tooltip>
          )}
        </NodeToolbar>

        {/* Handles for connecting edges */}
        <Handle type="source" id="t" position={Position.Top} />
        <Handle type="source" id="r" position={Position.Right} />

        {/* Box to display the speaker icon */}
        <Box
            sx={{
              mt: 1,
              fontSize: '1.5rem'
            }}
        >
          <SpeakerIcon sx={{ color: theme.palette.primary.main, size: '1em' }} />
        </Box>

        <Handle type="source" id="b" position={Position.Bottom} />
        <Handle type="source" id="l" position={Position.Left} />

        {/* Box to wrap the TextField */}
        <Box
            component="form"
            sx={{
              '& .MuiTextField-root': {
                width: '25ch',
                alignItems: 'center',
              }
            }}
            noValidate
            autoComplete="off"
        >
          {/* TextField for the node name */}
          <TextField
              inputRef={textRef}
              value={nodeName}
              onChange={handleChange}
              multiline
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
                  textAlign: 'center'
                },
              }}
          />
        </Box>
      </>
  );
}

export default memo(FlowNode);