import { Divider, Card, /*CardHeader,*/ Box, /*Grid,*/ Tabs, Tab } from '@material-ui/core'
//import useResizeObserver from 'use-resize-observer'
import ChatMessageList from 'src/components/_dashboard/chat/ChatMessageList'
import ChatMessageInput from 'src/components/_dashboard/chat/ChatMessageInput'
import { useState, useEffect } from 'react'
import InfoIcon from '@mui/icons-material/Info'
import IconButton from '@mui/material/IconButton'
import ToolTip from '@mui/material/Tooltip'
import { useVariantNotes } from '../../api/variant'

const useTriggerRerender = () => {
  const [time, setTime] = useState(Date.now())

  useEffect(() => {
    const interval = setInterval(() => setTime(Date.now()), 1000 * 60)
    return () => {
      clearInterval(interval)
    }
  }, [])

  return time
}

const NotesCard = function ({ variant, height, sampleName }) {
  const [tabPage, setTabPage] = useState(0)
  const { query, mutation } = useVariantNotes({ variantId: variant?.ID, sampleName })
  const { data = { messages: [], specificMessages: [] } } = query

  const specificData = { ...data }
  specificData.messages = data.specificMessages

  const rerenderTrigger = useTriggerRerender()

  const handleSendMessage = ({ message, isPrivate = false }) => {
    mutation.mutate({ note: message, isPrivate })
  }

  const handleTabChange = (event, newValue) => {
    setTabPage(newValue)
  }

  return (
    <Card>
      <Box sx={{ marginLeft: 1, borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabPage} onChange={handleTabChange} aria-label="basic tabs example">
          <Tab label="Notes" />
          <ToolTip title="Take a note for this variant within your organization">
            <IconButton>
              <InfoIcon />
            </IconButton>
          </ToolTip>
          <Tab label="Sample Specific Notes" />
          <ToolTip title={`Take a note for this variant only for ${sampleName}`}>
            <IconButton>
              <InfoIcon />
            </IconButton>
          </ToolTip>
        </Tabs>
      </Box>
      {/* TODO: Find a better way to set the height */}
      {tabPage === 0 && (
        <Box sx={{ height: height - 60 }}>
          <ChatMessageList conversation={data} rerenderTrigger={rerenderTrigger} />
        </Box>
      )}
      {tabPage === 1 && (
        <Box sx={{ height: height - 60 }}>
          <ChatMessageList conversation={specificData} rerenderTrigger={rerenderTrigger} />
        </Box>
      )}
      <Divider />
      <ChatMessageInput onSend={handleSendMessage} tabIndex={tabPage} />
    </Card>
  )

  // return (
  //   // <Card sx={{ maxHeight: 300, flexGrow: 1, display: 'flex', overflow: 'hidden', flexDirection: 'column' }}>
  //   //   <CardHeader title="Notes" />
  //   //   <Box sx={{ display: 'flex', flexGrow: 1, flexDirection: 'column' }}>
  //   //     <ChatMessageList conversation={data} />
  //   //     <Divider />
  //   //     <ChatMessageInput onSend={handleSendMessage} />
  //   //   </Box>
  //   // </Card>
  // )
}

export default NotesCard
