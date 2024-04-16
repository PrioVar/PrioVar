import { format, getTime, formatDistanceToNow, formatDistanceStrict } from 'date-fns'

// ----------------------------------------------------------------------

export function fDate(date) {
  return format(new Date(date), 'dd MMMM yyyy')
}

export function fDateTime(date) {
  return format(new Date(date), 'dd MMM yyyy HH:mm')
}

export function fTimestamp(date) {
  return getTime(new Date(date))
}

export function fDateTimeSuffix(date) {
  return format(new Date(date), 'dd/MM/yyyy hh:mm p')
}

export function fToNow(date) {
  return formatDistanceToNow(new Date(date), {
    addSuffix: true,
  })
}

export const fDuration = (seconds) => formatDistanceStrict(0, seconds * 1000, { roundingMethod: 'ceil' })
