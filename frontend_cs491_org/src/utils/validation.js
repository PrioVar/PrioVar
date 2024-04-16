import { isEmpty, isNil } from 'ramda'

export const isValid = (value) => !isEmpty(value) && !isNil(value)
