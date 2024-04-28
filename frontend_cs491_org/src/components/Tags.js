import React, { useCallback, useEffect, useState } from 'react'
import PropTypes from 'prop-types'
import { TextField, Autocomplete, Typography, ListSubheader, useMediaQuery } from '@material-ui/core'
import { useTheme } from '@material-ui/core/styles'
import parse from 'autosuggest-highlight/parse'
import match from 'autosuggest-highlight/match'
import { VariableSizeList } from 'react-window'
import { matchSorter } from 'match-sorter'

const LISTBOX_PADDING = 8 // px

function renderRow(props) {
  const { data, index, style } = props
  return React.cloneElement(data[index], {
    style: {
      ...style,
      top: style.top + LISTBOX_PADDING,
    },
  })
}

const OuterElementContext = React.createContext({})

const OuterElementType = React.forwardRef((props, ref) => {
  const outerProps = React.useContext(OuterElementContext)
  return <div ref={ref} {...props} {...outerProps} />
})

function useResetCache(data) {
  const ref = React.useRef(null)
  React.useEffect(() => {
    if (ref.current != null) {
      ref.current.resetAfterIndex(0, true)
    }
  }, [data])
  return ref
}

// Adapter for react-window
const ListboxComponent = React.forwardRef(function ListboxComponent(props, ref) {
  const { children, ...other } = props
  const itemData = React.Children.toArray(children)
  const theme = useTheme()
  const smUp = useMediaQuery(theme.breakpoints.up('sm'), { noSsr: true })
  const itemCount = itemData.length
  const itemSize = smUp ? 36 : 48

  const getChildSize = (child) => {
    if (React.isValidElement(child) && child.type === ListSubheader) {
      return 48
    }

    return itemSize
  }

  const getHeight = () => {
    if (itemCount > 8) {
      return 8 * itemSize
    }
    return itemData.map(getChildSize).reduce((a, b) => a + b, 0)
  }

  const gridRef = useResetCache(itemCount)

  return (
    <div ref={ref}>
      <OuterElementContext.Provider value={other}>
        <VariableSizeList
          itemData={itemData}
          height={getHeight() + 2 * LISTBOX_PADDING}
          width="100%"
          ref={gridRef}
          outerElementType={OuterElementType}
          innerElementType="ul"
          itemSize={(index) => getChildSize(itemData[index])}
          overscanCount={5}
          itemCount={itemCount}
        >
          {renderRow}
        </VariableSizeList>
      </OuterElementContext.Provider>
    </div>
  )
})
ListboxComponent.propTypes = {
  children: PropTypes.node,
}

function Tags({ options, title, value, onChange, freeSolo, sortKeys = ['value', 'label'], ...rest }) {
  const [inputValue, setInputValue] = useState('')

  const filterOptions = useCallback(
    (options, { inputValue }) => {
      const matches = matchSorter(options, inputValue, { keys: sortKeys })
      if (/*matches.length === 0 &&*/ freeSolo && inputValue !== '') {
        return [{ label: `Add "${inputValue}"`, value: inputValue, freeSolo: true }, ...matches]
      }
      return matches
    },
    [freeSolo],
  )

  const handleChange = (_event, newValueList) => {
    const newValuesFixed = newValueList
      .map((newValue) => {
        if (typeof newValue === 'string') {
          return { value: newValue, label: newValue }
        }
        if (newValue?.freeSolo) {
          return { label: newValue.value, value: newValue.value }
        }
        return newValue
      })
      .filter(Boolean)

    onChange(newValuesFixed)
  }

  useEffect(() => {
    setInputValue(value?.label ?? '')
  }, [value])

  return (
    <Autocomplete
      multiple
      selectOnFocus
      clearOnBlur
      freeSolo={freeSolo}
      ListboxComponent={ListboxComponent}
      inputValue={inputValue}
      value={value}
      onChange={handleChange}
      filterOptions={filterOptions}
      options={options}
      renderInput={(params) => (
        <TextField {...params} onChange={(e) => setInputValue(e.target.value)} label={title} variant="outlined" />
      )}
      renderOption={(props, option, state) => {
        const matches = match(option.label, state.inputValue)
        const parts = parse(option.label, matches)

        return (
          <li {...props}>
            {parts.map((part, index) => (
              <Typography
                key={index}
                noWrap
                style={{ fontWeight: part.highlight ? 800 : 500, whiteSpace: 'pre' }}
                component="span"
              >
                {part.text}
              </Typography>
            ))}
          </li>
        )
      }}
      {...rest}
    />
  )
}

Tags.propTypes = {
  title: PropTypes.string.isRequired,
  options: PropTypes.arrayOf(
    PropTypes.shape({ label: PropTypes.string.isRequired, value: PropTypes.string.isRequired }),
  ).isRequired,
  onChange: PropTypes.func.isRequired,
}

export default Tags
