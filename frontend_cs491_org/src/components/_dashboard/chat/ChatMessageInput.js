//import { v4 as uuidv4 } from 'uuid'
import PropTypes from 'prop-types'
import { Icon } from '@iconify/react'
import { useRef, useState } from 'react'
//import micFill from '@iconify/icons-eva/mic-fill'
import roundSend from '@iconify/icons-ic/round-send'
//import attach2Fill from '@iconify/icons-eva/attach-2-fill'
//import roundAddPhotoAlternate from '@iconify/icons-ic/round-add-photo-alternate'
// material
import { styled } from '@material-ui/core/styles'
import { Input, Divider, IconButton } from '@material-ui/core'
//
//import EmojiPicker from '../../EmojiPicker'

// ----------------------------------------------------------------------

const RootStyle = styled('div')(({ theme }) => ({
  minHeight: 56,
  display: 'flex',
  position: 'relative',
  alignItems: 'center',
  paddingLeft: theme.spacing(2),
}))

// ----------------------------------------------------------------------

ChatMessageInput.propTypes = {
  disabled: PropTypes.bool,
  conversationId: PropTypes.string,
  onSend: PropTypes.func,
}

export default function ChatMessageInput({ disabled, conversationId, onSend, ...other }) {
  const fileInputRef = useRef(null)
  const [message, setMessage] = useState('')

  const handleChangeMessage = (event) => {
    setMessage(event.target.value)
  }

  const handleKeyUp = (event) => {
    if (event.key === 'Enter') {
      const privateFlag = other.tabIndex === 1 ? true : false
      handleSend(privateFlag)
    }
  }

  const handleSend = (isPrivate = false) => {
    if (!message) {
      return ''
    }
    if (onSend) {
      onSend({ message, isPrivate })
    }
    return setMessage('')
  }

  return (
    <RootStyle {...other}>
      <Input
        disabled={disabled}
        fullWidth
        value={message}
        disableUnderline
        onKeyUp={handleKeyUp}
        onChange={handleChangeMessage}
        placeholder="Type a message"
        // startAdornment={
        //   <InputAdornment position="start">
        //     <EmojiPicker disabled={disabled} value={message} setValue={setMessage} />
        //   </InputAdornment>
        // }
        // endAdornment={
        //   <Stack direction="row" spacing={0.5} mr={1.5}>
        //     <IconButton disabled={disabled} size="small" onClick={handleAttach}>
        //       <Icon icon={roundAddPhotoAlternate} width={24} height={24} />
        //     </IconButton>
        //     <IconButton disabled={disabled} size="small" onClick={handleAttach}>
        //       <Icon icon={attach2Fill} width={24} height={24} />
        //     </IconButton>
        //     <IconButton disabled={disabled} size="small">
        //       <Icon icon={micFill} width={24} height={24} />
        //     </IconButton>
        //   </Stack>
        // }
        sx={{ height: '100%' }}
      />

      <Divider orientation="vertical" flexItem />

      <IconButton
        color="primary"
        disabled={!message}
        onClick={() => handleSend(other.tabIndex === 1 ? true : false)}
        sx={{ mx: 1 }}
      >
        <Icon icon={roundSend} width={24} height={24} />
      </IconButton>

      <input type="file" ref={fileInputRef} style={{ display: 'none' }} />
    </RootStyle>
  )
}

/*
const handleAttach = () => {
    fileInputRef.current.click()
  }
*/