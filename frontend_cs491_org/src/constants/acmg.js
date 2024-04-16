const createAcmgSeverityEntry = (value, description, order, color) => [value, { value, description, order, color }]

export const ACMG_SEVERITY = Object.fromEntries([
  createAcmgSeverityEntry('Uncertain significance', 'VUS', -1, 'warning'),
  createAcmgSeverityEntry('Benign', 'Benign', 0, 'default'),
  createAcmgSeverityEntry('Likely benign', 'Likely benign', 1, 'success'),
  createAcmgSeverityEntry('Likely pathogenic', 'Likely pathogenic', 2, 'warning'),
  createAcmgSeverityEntry('Pathogenic', 'Pathogenic', 3, 'error'),
])

export const ACMG_CHOICES = {
  DESCRIPTIONS: [
    { label: 'VUS', value: 'Uncertain significance' },
    { label: 'Benign', value: 'Benign' },
    { label: 'Likely benign', value: 'Likely benign' },
    { label: 'Likely pathogenic', value: 'Likely pathogenic' },
    { label: 'Pathogenic', value: 'Pathogenic' },
  ],
}
