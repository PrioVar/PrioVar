const createPvs1SeverityEntry = (value, description, order, color) => [value, { value, description, order, color }]

export const PVS1_SEVERITY = Object.fromEntries([
  createPvs1SeverityEntry('UNSET', 'Unset', -1, 'default'),
  createPvs1SeverityEntry('UNMET', 'VUS', 0, 'warning'),
  createPvs1SeverityEntry('SUPPORTING', 'Supporting', 1, 'success'),
  createPvs1SeverityEntry('MODERATE', 'Moderate', 2, 'warning'),
  createPvs1SeverityEntry('STRONG', 'Strong', 3, 'error'),
  createPvs1SeverityEntry('VERY_STRONG', 'Very Strong', 4, 'error'),
])

const createPvs1CriteriaEntry = (value, severity) => [value, { criteria: value, severity }]

const pvs1CriteriaMap = Object.fromEntries([
  createPvs1CriteriaEntry('CDH1', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('IC0', PVS1_SEVERITY.VERY_STRONG),
  createPvs1CriteriaEntry('IC1', PVS1_SEVERITY.VERY_STRONG),
  createPvs1CriteriaEntry('IC2', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('IC3', PVS1_SEVERITY.MODERATE),
  createPvs1CriteriaEntry('IC4', PVS1_SEVERITY.SUPPORTING),
  createPvs1CriteriaEntry('IC5', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('NF0', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('NF1', PVS1_SEVERITY.VERY_STRONG),
  createPvs1CriteriaEntry('NF2', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('NF3', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('NF4', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('NF5', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('NF6', PVS1_SEVERITY.MODERATE),
  createPvs1CriteriaEntry('PTEN', PVS1_SEVERITY.VERY_STRONG),
  createPvs1CriteriaEntry('SS1', PVS1_SEVERITY.VERY_STRONG),
  createPvs1CriteriaEntry('SS10', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('SS2', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('SS3', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('SS4', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('SS5', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('SS6', PVS1_SEVERITY.MODERATE),
  createPvs1CriteriaEntry('SS7', PVS1_SEVERITY.UNMET),
  createPvs1CriteriaEntry('SS8', PVS1_SEVERITY.STRONG),
  createPvs1CriteriaEntry('SS9', PVS1_SEVERITY.MODERATE),
])

export const PVS1_CRITERIA = {
  ...pvs1CriteriaMap,

  fromString(string) {
    return pvs1CriteriaMap[string]
  },
}
