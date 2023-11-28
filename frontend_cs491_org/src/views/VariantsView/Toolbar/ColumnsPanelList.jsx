import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  FormControlLabel,
  List,
  ListItem,
  Switch,
  Typography,
} from '@material-ui/core'
import { withStyles } from '@material-ui/styles'
import DragHandleIcon from '@material-ui/icons/DragHandle'
import PropTypes from 'prop-types'
import React from 'react'
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd'

import ViewColumnIcon from '@material-ui/icons/ViewColumn'

const CPLAccordion = withStyles({
  root: {
    backgroundColor: '#00ab55',
    border: '1px solid rgba(0, 0, 0, .125)',
  },
  expanded: {},
})(Accordion)

const CPLAccordionSummary = withStyles({
  root: {
    borderBottom: '1px solid rgba(0, 0, 0, .125)',
    marginBottom: -1,
    '&$expanded': {
      minHeight: 56,
    },
  },
  content: {
    '&$expanded': {
      margin: '12px 0',
    },
  },
  expanded: {},
})(AccordionSummary)

const CPLAccordionDetails = withStyles((theme) => ({
  root: {
    backgroundColor: theme.palette.background.default,
    padding: theme.spacing(2),
  },
}))(AccordionDetails)

function DraggableListItem({ id, index, children, ...rest }) {
  return (
    <Draggable draggableId={id} index={index}>
      {(provided, snapshot) => (
        <ListItem ref={provided.innerRef} {...provided.draggableProps} selected={snapshot.isDragging} {...rest}>
          <Box {...provided.dragHandleProps} display="flex" ml={-0.5} pr={0.75}>
            <DragHandleIcon />
          </Box>
          {children}
        </ListItem>
      )}
    </Draggable>
  )
}

function ColumnsPanelList({ columns, onColumnHide, onColumnReorder, ...rest }) {
  const handleDragEnd = ({ destination, source }) => {
    // dropped outside the list
    if (!destination || !source) return

    onColumnReorder(source.index, destination.index)
  }

  return (
    <CPLAccordion id="filterSet">
      <CPLAccordionSummary expandIcon={<ViewColumnIcon />} aria-controls="panel1a-content" id="panel1a-header">
        <Typography variant="subtitle2">Columns</Typography>
      </CPLAccordionSummary>
      <CPLAccordionDetails>
        <List {...rest}>
          <DragDropContext onDragEnd={handleDragEnd}>
            <Droppable droppableId="droppable-columns">
              {(provided) => (
                <div ref={provided.innerRef} {...provided.droppableProps}>
                  {columns.map((column, cIndex) => (
                    <DraggableListItem id={column.label} index={cIndex} key={column.label} divider disableRipple>
                      <FormControlLabel
                        label={column.label}
                        control={<Switch checked={!column.hidden} onChange={() => onColumnHide(column, cIndex)} />}
                      />
                    </DraggableListItem>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>
        </List>
      </CPLAccordionDetails>
    </CPLAccordion>
  )
}

ColumnsPanelList.propTypes = {
  columns: PropTypes.array.isRequired,
  onColumnHide: PropTypes.func.isRequired,
  onColumnReorder: PropTypes.func.isRequired,
}

export default ColumnsPanelList
