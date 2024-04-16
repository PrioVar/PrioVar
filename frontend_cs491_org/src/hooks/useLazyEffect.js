import { useEffect, useRef } from 'react'

function useLazyEffect(cb, dep) {
  const initializeRef = useRef(false)

  useEffect((...args) => {
    if (initializeRef.current) {
      cb(...args)
    } else {
      initializeRef.current = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dep)
}

export default useLazyEffect
