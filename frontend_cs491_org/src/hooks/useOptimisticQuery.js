import { useEffect, useState, useRef } from 'react'
import { useMutation, useQuery } from 'react-query'
import useDebouncedValue from './useDebouncedValue'

const useOptimisticQuery = ({ queryFn, mutationFn, queryKey, delay }) => {
  const [value, setValue] = useState()
  const debouncedValue = useDebouncedValue(value, delay)
  const isFirstDebounce = useRef(true)
  const query = useQuery(queryKey, queryFn)
  const mutation = useMutation(mutationFn)

  useEffect(() => {
    if (query.status === 'success') {
      setValue(query.data)
    }
  }, [query.status])

  useEffect(() => {
    if (query.status === 'success') {
      if (isFirstDebounce.current) {
        isFirstDebounce.current = false
      } else {
        mutation.mutate(debouncedValue)
      }
    }
  }, [debouncedValue])

  return [value, setValue]
}

export default useOptimisticQuery
