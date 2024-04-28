import { useMutation, useQuery, useQueryClient } from 'react-query'
import { useContext } from 'react'
import { AuthContext } from '../../contexts/TokenContext'
import axios from '../../utils/axios'
import { API_BASE_URL } from '../../constants'

const mapToMinimalsContact = (note) => {
  const { user } = note.person

  return {
    id: user.id,
    // avatar: '/static/mock-images/avatars/avatar_15.jpg',
    name: `${user.first_name} ${user.last_name}`,
    username: user.username,
  }
}
const mapToMinimalsMessage = (note) => {
  const { user } = note.person
  return {
    id: note.id,
    body: note.text,
    contentType: 'text',
    attachments: [],
    createdAt: new Date(note.created),
    senderId: user.id,
  }
}
const mapToMinimalsConversation = (variantId, notes, specificNotes) => {
  const id = variantId
  const participants = notes.map(mapToMinimalsContact)
  const type = 'GROUP'
  const unreadCount = 0
  const messages = notes.map(mapToMinimalsMessage)
  const specificMessages = specificNotes.map(mapToMinimalsMessage)

  return { id, participants, type, unreadCount, messages, specificMessages }
}
const populateSelfMessages = (userId, notes) => {
  return notes.map((note) => ({
    ...note,
    person: {
      ...note.person,
      user: {
        ...note.person.user,
        // eslint-disable-next-line eqeqeq
        id: userId == note.person.user.id ? 'me' : note.person.user.id,
      },
    },
  }))
}
const fetchVariantNotes = async (userId, variantId, sampleName) => {
  if (!variantId) {
    return undefined
  }
  const { data } = await axios.get(`${API_BASE_URL}/variant/${variantId}/notes`)
  const { data: specificData } = await axios.get(`${API_BASE_URL}/variant/${variantId}/${sampleName}/notes`)
  const notes = populateSelfMessages(userId, data)
  const specificNotes = populateSelfMessages(userId, specificData)
  return mapToMinimalsConversation(variantId, notes, specificNotes)
}
const insertVariantNote = async (variantId, { note, isPrivate }, sampleName = '') => {
  const { data } = !isPrivate
    ? await axios.post(`${API_BASE_URL}/variant/${variantId}/notes`, {
        text: note,
      })
    : await axios.post(`${API_BASE_URL}/variant/${variantId}/${sampleName}/notes`, {
        text: note,
      })

  return data
}
export const useVariantNotes = ({ variantId, sampleName }) => {
  const queryClient = useQueryClient()
  const { user } = useContext(AuthContext)

  const queryKey = ['variant-notes', user.id, variantId, sampleName]
  const query = useQuery(queryKey, () => fetchVariantNotes(user.id, variantId, sampleName))
  const mutation = useMutation(({ note, isPrivate }) => insertVariantNote(variantId, { note, isPrivate }, sampleName), {
    onSuccess: () => queryClient.invalidateQueries(queryKey),
  })

  return { query, mutation }
}
