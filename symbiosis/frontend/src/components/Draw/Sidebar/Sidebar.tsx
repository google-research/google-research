import { DragEvent } from 'react';
import DataUsageIcon from '@mui/icons-material/DataUsage';
import SpeakerIcon from '@mui/icons-material/Speaker';
import InboxIcon from '@mui/icons-material/Inbox';

import { useTheme } from '@mui/material/styles';

// Sidebar component for dragging and dropping nodes into the React Flow diagram
const Sidebar = () => {
  const theme = useTheme(); // Access the theme to get dynamic colors

  // Function to handle drag start events for the draggable nodes
  const onDragStart = (event: DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType); // Set the node type in the drag data
    event.dataTransfer.effectAllowed = 'move'; // Allow the node to be moved
  };

  return (
      <aside> {/* Wrap the sidebar content in an aside element */}
        {/* Variable node */}
        <div
            className="react-flow__node-variablenode"
            onDragStart={(event: DragEvent) => onDragStart(event, 'variablenode')}
            draggable
        >
          <DataUsageIcon sx={{
            color: theme.palette.primary.main, // Set the icon color using the theme
            size: '1em',
            mt: '0.3em'
          }} />
          <div className="variablenodelabel">Variable</div> {/* Label for the variable node */}
        </div>

        {/* Stock node */}
        <div
            className="react-flow__node-stocknode"
            onDragStart={(event: DragEvent) => onDragStart(event, 'stocknode')}
            draggable
        >
          <InboxIcon sx={{
            color: theme.palette.primary.main, // Set the icon color using the theme
            size: '1em',
            mt: '0.3em',
            mb: '0.3em',
          }} />
          <div>Stock</div> {/* Label for the stock node */}
        </div>

        <br /> {/* Add some space between the nodes */}

        {/* Flow node */}
        <div
            className="react-flow__node-flownode"
            onDragStart={(event: DragEvent) => onDragStart(event, 'flownode')}
            draggable
        >
          <SpeakerIcon sx={{
            color: theme.palette.primary.main, // Set the icon color using the theme
            size: '1em',
            mt: '0.3em',
            mb: '0.3em',
          }} />
          <div>Flow</div> {/* Label for the flow node */}
        </div>

        <br /> {/* Add some space between the nodes */}

        {/* Group node */}
        <div
            className="react-flow__node-group"
            onDragStart={(event: DragEvent) => onDragStart(event, 'group')}
            draggable
        >
          <div className="variablenodelabel">Group</div> {/* Label for the group node */}
        </div>
      </aside>
  );
};

export default Sidebar;