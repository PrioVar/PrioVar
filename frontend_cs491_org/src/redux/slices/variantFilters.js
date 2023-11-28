import { createSlice } from '@reduxjs/toolkit'

const createVariantFilter = (initialValue, backendName, enabled = false) => ({
  initialValue,
  backendName,
  value: initialValue,
  enabled,
})

const initialState = {
  af: createVariantFilter([0, 1], 'AfRowFilter'),
  cadd: createVariantFilter([0, 1], 'CaddRowFilter'),
  dann: createVariantFilter([0, 1], 'DannRowFilter'),
  metalR: createVariantFilter([0, 1], 'MetalRRowFilter'),
  chromFilter: createVariantFilter('', 'ChromRowFilter'),
  fsFilter: createVariantFilter([0, 1000], 'FsRowFilter'),
  gnomAdFrequency: createVariantFilter([0, 1], 'GnomAdAfRowFilter'),
  exAcAf: createVariantFilter([0, 1], 'ExAcAfRowFilter'),
  turkishVariomeAf: createVariantFilter([0, 1], 'TurkishVariomeAfRowFilter'),
  impacts: createVariantFilter([], 'ImpactsRowFilter', true),
  inDbSnp: createVariantFilter('ANY', 'DbsnpRowFilter', true),
  oneKgFrequency: createVariantFilter([0, 1], 'OneKgRowFilter'),
  polyphen: createVariantFilter([0, 1], 'PolyphenRowFilter'),
  posFilter: createVariantFilter('', 'PosRowFilter'),
  qualFilter: createVariantFilter([0, 150000], 'QualRowFilter'),
  scenarioFilter: createVariantFilter('NONE', 'scenario'),
  sift: createVariantFilter([0, 1], 'SiftRowFilter'),
  kh1: createVariantFilter(false, 'Kh1RowFilter', true),
  multiposFilter: createVariantFilter('', 'MultiposRowFilter', true),
  kh3: createVariantFilter(false, 'OmimRowFilter', true),
  kh2: createVariantFilter(false, 'Kh2RowFilter', true),
  filter: createVariantFilter('ANY', 'FilterRowFilter', true),
  gt: createVariantFilter([], 'GtRowFilter', true),
  geneNames: createVariantFilter([''], 'GeneNameRowFilter'),
  pvs1: createVariantFilter([], 'Pvs1RowFilter'),
  hpoFilter: createVariantFilter(null, 'HpoRowFilter', true),
  predefinedFilter: createVariantFilter(null, 'PredefinedRowFilter', true),
  nonBenignFilter: createVariantFilter(false, 'NonBenignRowFilter', true),
  acmg: createVariantFilter([], 'AcmgRowFilter', true),
  clinVar: createVariantFilter([], 'ClinvarRowFilter', true),
  phenotypes: createVariantFilter([''], 'DiseasesRowFilter', true),
  dp: createVariantFilter([0, 1000], 'DpRowFilter'),
  ab: createVariantFilter([0, 1000], 'AbRowFilter'),
}

const slice = createSlice({
  name: 'variantFilters',
  initialState,
  reducers: {
    setAf(state, action) {
      state.af.value = action.payload
    },
    setCadd(state, action) {
      state.cadd.value = action.payload
    },
    setDann(state, action) {
      state.dann.value = action.payload
    },
    setMetalR(state, action) {
      state.metalR.value = action.payload
    },
    setChrom(state, action) {
      state.chromFilter.value = action.payload
    },
    setFs(state, action) {
      state.fsFilter.value = action.payload
    },
    setGnomAdFrequency(state, action) {
      state.gnomAdFrequency.value = action.payload
    },
    setExAcAf(state, action) {
      state.exAcAf.value = action.payload
    },
    setTurkishVariomeAf(state, action) {
      state.turkishVariomeAf.value = action.payload
    },
    setImpacts(state, action) {
      state.impacts.value = action.payload
    },
    setInDbSnp(state, action) {
      state.inDbSnp.value = action.payload
    },
    setOneKgFrequency(state, action) {
      state.oneKgFrequency.value = action.payload
    },
    setPolyphen(state, action) {
      state.polyphen.value = action.payload
    },
    setPos(state, action) {
      state.posFilter.value = action.payload
    },
    setQual(state, action) {
      state.qualFilter.value = action.payload
    },
    setScenario(state, action) {
      state.scenarioFilter.value = action.payload
    },
    setSift(state, action) {
      state.sift.value = action.payload
    },
    setClinVar(state, action) {
      state.clinVar.value = action.payload
    },
    setKh1(state, action) {
      state.kh1.value = action.payload
    },
    setMultipos(state, action) {
      state.multiposFilter.value = action.payload
    },
    setHpo(state, action) {
      state.hpoFilter.value = action.payload
    },
    setPredefined(state, action) {
      state.predefinedFilter.value = action.payload
    },
    setNonBenign(state, action) {
      state.nonBenignFilter.value = action.payload
    },
    setKh3(state, action) {
      state.kh3.value = action.payload
    },
    setKh2(state, action) {
      state.kh2.value = action.payload
    },
    setFilter(state, action) {
      state.filter.value = action.payload
    },
    setGt(state, action) {
      state.gt.value = action.payload
    },
    setGeneNames(state, action) {
      state.geneNames.value = action.payload
    },
    setAcmg(state, action) {
      state.acmg.value = action.payload
    },
    setPhenotype(state, action) {
      state.phenotypes.value = action.payload
    },
    setDp(state, action) {
      state.dp.value = action.payload
    },
    setAb(state, action) {
      state.ab.value = action.payload
    },
    toggleAf(state, action) {
      state.af.enabled = !state.af.enabled
    },
    toggleCadd(state, action) {
      state.cadd.enabled = !state.cadd.enabled
    },
    toggleDann(state, action) {
      state.dann.enabled = !state.dann.enabled
    },
    toggleMetalR(state, action) {
      state.metalR.enabled = !state.metalR.enabled
    },
    toggleChrom(state, action) {
      state.chromFilter.enabled = !state.chromFilter.enabled
    },
    toggleFs(state, action) {
      state.fsFilter.enabled = !state.fsFilter.enabled
    },
    toggleGnomAdFrequency(state, action) {
      state.gnomAdFrequency.enabled = !state.gnomAdFrequency.enabled
    },
    toggleExAcAf(state, action) {
      state.exAcAf.enabled = !state.exAcAf.enabled
    },
    toggleTurkishVariomeAf(state, action) {
      state.turkishVariomeAf.enabled = !state.turkishVariomeAf.enabled
    },
    toggleImpacts(state, action) {
      state.impacts.enabled = !state.impacts.enabled
    },
    toggleInDbSnp(state, action) {
      state.inDbSnp.enabled = !state.inDbSnp.enabled
    },
    toggleOneKgFrequency(state, action) {
      state.oneKgFrequency.enabled = !state.oneKgFrequency.enabled
    },
    togglePolyphen(state, action) {
      state.polyphen.enabled = !state.polyphen.enabled
    },
    togglePos(state, action) {
      state.posFilter.enabled = !state.posFilter.enabled
    },
    toggleQual(state, action) {
      state.qualFilter.enabled = !state.qualFilter.enabled
    },
    toggleScenario(state, action) {
      state.scenarioFilter.enabled = !state.scenarioFilter.enabled
    },
    toggleSift(state, action) {
      state.sift.enabled = !state.sift.enabled
    },
    toggleClinVar(state, action) {
      state.clinVar.enabled = !state.clinVar.enabled
    },
    toggleKh1(state, action) {
      state.kh1.enabled = !state.kh1.enabled
    },
    toggleMultipos(state, action) {
      state.multiposFilter.enabled = !state.multiposFilter.enabled
    },
    toggleHpo(state, action) {
      state.hpoFilter.enabled = !state.hpoFilter.enabled
    },
    togglePredefined(state, action) {
      state.predefinedFilter.enabled = !state.predefinedFilter.enabled
    },
    toggleNonBenign(state, action) {
      state.nonBenignFilter.enabled = !state.predefinedFilter.enabled
    },
    toggleKh3(state, action) {
      state.kh3.enabled = !state.kh3.enabled
    },
    toggleKh2(state, action) {
      state.kh2.enabled = !state.kh2.enabled
    },
    toggleFilter(state, action) {
      state.filter.enabled = !state.filter.enabled
    },
    toggleGt(state, action) {
      state.gt.enabled = !state.gt.enabled
    },
    toggleGeneNames(state, action) {
      state.geneNames.enabled = !state.geneNames.enabled
    },
    toggleAcmg(state, action) {
      state.acmg.enabled = !state.acmg.enabled
    },
    toggleDp(state, action) {
      state.dp.enabled = !state.dp.enabled
    },
    toggleAb(state, action) {
      state.ab.enabled = !state.ab.enabled
    },
    setSavedFilters(state, action) {
      state = { ...action.payload }
    },
  },
})

export const reducer = slice.reducer
export const actions = slice.actions
