import { Box, FormControlLabel, Menu, MenuItem, Switch } from '@material-ui/core'
import DragHandleIcon from '@material-ui/icons/DragHandle'
import PropTypes from 'prop-types'
import React from 'react'
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd'

function DraggableMenuItem({ id, index, children, ...rest }) {
  return (
    <Draggable draggableId={id} index={index}>
      {(provided, snapshot) => (
        <MenuItem ref={provided.innerRef} {...provided.draggableProps} selected={snapshot.isDragging} {...rest}>
          <Box {...provided.dragHandleProps} display="flex" ml={-0.5} pr={0.75}>
            <DragHandleIcon />
          </Box>
          {children}
        </MenuItem>
      )}
    </Draggable>
  )
}

function ColumnsPanelMenu({ columns, onColumnHide, onColumnReorder, ...rest }) {
  const handleDragEnd = ({ destination, source }) => {
    // dropped outside the list
    if (!destination || !source) return

    onColumnReorder(source.index, destination.index)
  }

  return (
    <Menu {...rest}>
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="droppable-columns">
          {(provided) => (
            <div ref={provided.innerRef} {...provided.droppableProps}>
              {columns.map((column, cIndex) => (
                <DraggableMenuItem id={column.label} index={cIndex} key={column.label} divider disableRipple>
                  <FormControlLabel
                    label={column.label}
                    control={<Switch checked={!column.hidden} onChange={() => onColumnHide(column, cIndex)} />}
                  />
                </DraggableMenuItem>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    </Menu>
  )
}

ColumnsPanelMenu.propTypes = {
  columns: PropTypes.array.isRequired,
  onColumnHide: PropTypes.func.isRequired,
  onColumnReorder: PropTypes.func.isRequired,
}

export default ColumnsPanelMenu
