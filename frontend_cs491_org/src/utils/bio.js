import { countBy } from 'lodash'
import { maxBy } from 'ramda'
import { IMPACT_CHOICES_MAP, PVS1_CRITERIA, SEVERITY_COLOR_ORDER } from 'src/constants'

export const isHomogeneous = (value) => {
  const matches = value.match(/\d+/g)
  return matches.every((match) => match === matches[0])
}
export const groupImpacts = (values) => {
  const low = values.map((v) => IMPACT_CHOICES_MAP.LOW[v]).filter(Boolean)
  const moderate = values.map((v) => IMPACT_CHOICES_MAP.MODERATE[v]).filter(Boolean)
  const high = values.map((v) => IMPACT_CHOICES_MAP.HIGH[v]).filter(Boolean)
  const other = values
    .filter((v) => !IMPACT_CHOICES_MAP.LOW[v] && !IMPACT_CHOICES_MAP.MODERATE[v] && !IMPACT_CHOICES_MAP.HIGH[v])
    .map((v) => v.replace('_variant', '').replace(/_/g, ' '))
    .map((value) => ({ label: value, value }))

  return { low, moderate, high, other }
}

export const getImpactDescription = (labels) => {
  const countMap = countBy(labels)

  return Object.entries(countMap).map(([label, count]) => `${count} ✖ ${label}`)
}

const maxByPvs1Severity = maxBy((pvs1) => pvs1.severity.order)

export const getMostSeverePvs1 = (rawPvs1Criterias) => {
  const pvs1Criterias = rawPvs1Criterias.map(PVS1_CRITERIA.fromString)
  const mostSeverePvs1 = pvs1Criterias.reduce(maxByPvs1Severity, { severity: { order: -100 } })

  return mostSeverePvs1
}

const maxBySeverityColor = maxBy((color) => SEVERITY_COLOR_ORDER.indexOf(color))

export const getMostSevereColor = (colors) => {
  return colors.reduce(maxBySeverityColor, 'default')
}
