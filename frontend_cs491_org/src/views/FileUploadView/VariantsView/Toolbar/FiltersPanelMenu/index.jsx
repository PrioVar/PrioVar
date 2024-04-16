import { Box, Button, CardContent } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import React, { useState, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { hpoIdtoGeneName } from 'src/api/gene'

import GenotypeFilter from 'src/views/VariantsView/Toolbar/FiltersPanelMenu/GenotypeFilter'

import FiltersAccordion from './FiltersAccordion'
import FrequencyFilters from './FrequencyFilters'
import ImpactFilters from './ImpactFilters'
import LocationFilters from './LocationFilters'
import PathogenicityFilters from './PathogenicityFilters'
import QualityFilters from './QualityFilters'
import GeneSetFilters from './GeneSetFilters'
import PhenotypeFilter from './PhenotypeFilter'
import HPOFilter from './HPOFilter'
import ReadFilters from './ReadFilters'

import { actions as variantFiltersActions } from 'src/redux/slices/variantFilters'
import SavedFilters from './SavedFiltersFilters'

const useStyles = makeStyles({
  cardContent: {
    padding: 0,
  },
})

const geneNamesInputTextToList = (text) => {
  return text.replace(/\s+/g, '').split(',')
}

function FiltersPanelMenu(props) {
  const classes = useStyles()
  const dispatch = useDispatch()

  // location filter states
  const [chromEvent, setChromEvent] = useState()
  const [posEvent, setPosEvent] = useState()
  const [geneNamesInputText, setGeneNamesInputText] = useState('')
  const [listOfGeneNames, setListOfGeneNames] = useState('')
  const state = useSelector((state) => state)

  const [toggleSelectorChrom, setToggleSelectorChrom] = useState(false)
  const [toggleSelectorPos, setToggleSelectorPos] = useState(false)
  const [toggleSelectorGene, setToggleSelectorGene] = useState(false)
  const [isToggleChangedLoc, setIsToggleChangedLoc] = useState([false, false, false])

  // location filter functions
  const handleChromChange = (event) => {
    dispatch(variantFiltersActions.setChrom(event || ''))
  }

  const handleChromToggle = () => {
    isToggleChangedLoc[0] && dispatch(variantFiltersActions.toggleChrom())
    isToggleChangedLoc[0] && setIsToggleChangedLoc([false, isToggleChangedLoc[1], isToggleChangedLoc[2]])
  }

  const handlePosChange = (event) => {
    dispatch(variantFiltersActions.setPos(event || ''))
  }

  const handlePosToggle = () => {
    isToggleChangedLoc[1] && toggleSelectorPos && dispatch(variantFiltersActions.togglePos())
    isToggleChangedLoc[1] && setIsToggleChangedLoc([isToggleChangedLoc[0], false, isToggleChangedLoc[2]])
  }

  const handleGeneNamesChange = () => {
    dispatch(variantFiltersActions.setGeneNames(listOfGeneNames))
  }

  const handleGeneNamesToggle = () => {
    isToggleChangedLoc[2] && toggleSelectorGene && dispatch(variantFiltersActions.toggleGeneNames())
    isToggleChangedLoc[2] && setIsToggleChangedLoc([isToggleChangedLoc[0], isToggleChangedLoc[1], false])
  }

  useEffect(() => {
    setListOfGeneNames(geneNamesInputTextToList(geneNamesInputText))
  }, [dispatch, geneNamesInputText])

  // genotype filter states
  const gtFilter = useSelector((state) => state.gt)
  const [gtFilterEvent, setGtFilterEvent] = useState(gtFilter.value ? gtFilter.value : 'all')

  // genotype filter functions
  const handleGtFilterChange = (value) => {
    dispatch(variantFiltersActions.setGt(value))
  }

  // frequency filter states
  const inDbSnp = useSelector((state) => state.inDbSnp)
  const af = useSelector((state) => state.af)
  const oneKgFrequency = useSelector((state) => state.oneKgFrequency)
  const gnomAdFrequency = useSelector((state) => state.gnomAdFrequency)
  const exAcAf = useSelector((state) => state.exAcAf)
  const turkishVariomeAf = useSelector((state) => state.turkishVariomeAf)

  const [afEvent, setAfEvent] = useState(af.value)
  const [oneKgFrequencyEvent, setOneKgFrequencyEvent] = useState(oneKgFrequency.value)
  const [gnomAdFrequencyEvent, setGnomAdFrequencyEvent] = useState(gnomAdFrequency.value)
  const [exAcAfEvent, setExAcAfEvent] = useState(exAcAf.value)
  const [turkishVariomeAfEvent, setTurkishVariomeAfEvent] = useState(turkishVariomeAf.value)
  const [inDbsnpEvent, setInDbsnpEvent] = useState(inDbSnp.value) // buttons

  const [toggleSelectorAf, setToggleSelectorAf] = useState(false)
  const [toggleSelectorOneKgFrequency, setToggleSelectorOneKgFrequency] = useState(false)
  const [toggleSelectorGnomAdFrequency, setToggleSelectorGnomAdFrequency] = useState(false)
  const [toggleSelectorExAcAf, setToggleSelectorExAcAf] = useState(false)
  const [toggleSelectorTurkishVariomeAf, setToggleSelectorTurkishVariomeAf] = useState(false)
  const [isToggleChangedFrequency, setIsToggleChangedFrequency] = useState([false, false, false, false, false])

  // frequency filter functions
  const handleInDbsnpChange = (value) => {
    dispatch(variantFiltersActions.setInDbSnp(value))
  }

  const handleAfChange = (value) => {
    dispatch(variantFiltersActions.setAf(value))
  }

  const handleAfToggle = () => {
    isToggleChangedFrequency[0] && dispatch(variantFiltersActions.toggleAf())
    isToggleChangedFrequency[0] &&
      setIsToggleChangedFrequency([
        false,
        isToggleChangedFrequency[1],
        isToggleChangedFrequency[2],
        isToggleChangedFrequency[3],
        isToggleChangedFrequency[4],
      ])
  }

  const handleOneKgChange = (value) => {
    dispatch(variantFiltersActions.setOneKgFrequency(value))
  }

  const handleOneKgToggle = () => {
    isToggleChangedFrequency[1] && dispatch(variantFiltersActions.toggleOneKgFrequency())
    isToggleChangedFrequency[1] &&
      setIsToggleChangedFrequency([
        isToggleChangedFrequency[0],
        false,
        isToggleChangedFrequency[2],
        isToggleChangedFrequency[3],
        isToggleChangedFrequency[4],
      ])
  }

  const handleGnomAdFrequencyChange = (value) => {
    dispatch(variantFiltersActions.setGnomAdFrequency(value))
  }

  const handleGnomAdFrequencyToggle = () => {
    isToggleChangedFrequency[2] && dispatch(variantFiltersActions.toggleGnomAdFrequency())
    isToggleChangedFrequency[2] &&
      setIsToggleChangedFrequency([
        isToggleChangedFrequency[0],
        isToggleChangedFrequency[1],
        false,
        isToggleChangedFrequency[3],
        isToggleChangedFrequency[4],
      ])
  }

  const handleExAcAfChange = (value) => {
    dispatch(variantFiltersActions.setExAcAf(value))
  }

  const handleExAcAfToggle = () => {
    isToggleChangedFrequency[3] && dispatch(variantFiltersActions.toggleExAcAf())
    isToggleChangedFrequency[3] &&
      setIsToggleChangedFrequency([
        isToggleChangedFrequency[0],
        isToggleChangedFrequency[1],
        isToggleChangedFrequency[2],
        false,
        isToggleChangedFrequency[4],
      ])
  }

  const handleTurkishVariomeAfChange = (value) => {
    dispatch(variantFiltersActions.setTurkishVariomeAf(value))
  }

  const handleTurkishVariomeAfToggle = () => {
    isToggleChangedFrequency[4] && dispatch(variantFiltersActions.toggleTurkishVariomeAf())
    isToggleChangedFrequency[4] &&
      setIsToggleChangedFrequency([
        isToggleChangedFrequency[0],
        isToggleChangedFrequency[1],
        isToggleChangedFrequency[2],
        isToggleChangedFrequency[3],
        false,
      ])
  }

  // quality filter states
  const qualFilter = useSelector((state) => state.qualFilter)
  const fsFilter = useSelector((state) => state.fsFilter)
  const filterFilter = useSelector((state) => state.filter)

  const [qualFEvent, setQuallFEvent] = useState(qualFilter.value)
  const [fsFEvent, setFsFEvent] = useState(fsFilter.value)
  const [filterFEvent, setfilterFEvent] = useState(filterFilter.value)

  const [toggleSelectorQualF, setToggleSelectorQualF] = useState(false)
  const [toggleSelectorFsF, setToggleSelectorFsF] = useState(false)
  const [isToggleChangedQuality, setIsToggleChangedQuality] = useState([false, false])

  // quality filter functions
  const handleQualFChange = (value) => {
    dispatch(variantFiltersActions.setQual(value))
  }

  const handleQualFToggle = () => {
    isToggleChangedQuality[0] && dispatch(variantFiltersActions.toggleQual())
    isToggleChangedQuality[0] && setIsToggleChangedQuality([false, isToggleChangedQuality[1]])
  }

  const handleFsFChange = (value) => {
    dispatch(variantFiltersActions.setFs(value))
  }

  const handleFsFToggle = () => {
    isToggleChangedQuality[1] && dispatch(variantFiltersActions.toggleFs())
    isToggleChangedQuality[1] && setIsToggleChangedQuality([isToggleChangedQuality[0], false])
  }

  const handleFilterFChange = (value) => {
    dispatch(variantFiltersActions.setFilter(value))
  }

  // impact filter states
  const [impactEvent, setImpactEvent] = useState([])

  // impact filter functions
  const handleImpactEventChange = () => {
    dispatch(variantFiltersActions.setImpacts(impactEvent))
  }

  // pathogenicity filter states
  const cadd = useSelector((state) => state.cadd)
  const dann = useSelector((state) => state.dann)
  const metalR = useSelector((state) => state.metalR)
  const polyphen = useSelector((state) => state.polyphen)
  const sift = useSelector((state) => state.sift)

  const [caddEvent, setCaddEvent] = useState(cadd.value)
  const [dannEvent, setDannEvent] = useState(dann.value)
  const [metalREvent, setMetalREvent] = useState(metalR.value)
  const [polyEvent, setPolyEvent] = useState(polyphen.value)
  const [siftEvent, setSiftEvent] = useState(sift.value)

  const [toggleSelectorCadd, setToggleSelectorCadd] = useState(false)
  const [toggleSelectorDann, setToggleSelectorDann] = useState(false)
  const [toggleSelectorMetalR, setToggleSelectorMetalR] = useState(false)
  const [toggleSelectorPoly, setToggleSelectorPoly] = useState(false)
  const [toggleSelectorSift, setToggleSelectorSift] = useState(false)
  const [isToggleChangedPatho, setIsToggleChangedPatho] = useState([false, false, false, false, false, false])

  // pathogenicity filter functions
  const handleCaddChange = (value) => {
    dispatch(variantFiltersActions.setCadd(value))
  }

  const handleCaddToggle = () => {
    isToggleChangedPatho[0] && dispatch(variantFiltersActions.toggleCadd())
    isToggleChangedPatho[0] &&
      setIsToggleChangedPatho([
        false,
        isToggleChangedLoc[1],
        isToggleChangedLoc[2],
        isToggleChangedLoc[3],
        isToggleChangedLoc[4],
        isToggleChangedLoc[5],
      ])
  }

  const handleDannChange = (value) => {
    dispatch(variantFiltersActions.setDann(value))
  }

  const handleDannToggle = () => {
    isToggleChangedPatho[1] && dispatch(variantFiltersActions.toggleDann())
    isToggleChangedPatho[1] &&
      setIsToggleChangedPatho([
        isToggleChangedLoc[0],
        false,
        isToggleChangedLoc[2],
        isToggleChangedLoc[3],
        isToggleChangedLoc[4],
        isToggleChangedLoc[5],
      ])
  }

  const handleMetalRChange = (value) => {
    dispatch(variantFiltersActions.setMetalR(value))
  }

  const handleMetalRToggle = () => {
    isToggleChangedPatho[2] && dispatch(variantFiltersActions.toggleMetalR())
    isToggleChangedPatho[2] &&
      setIsToggleChangedPatho([
        isToggleChangedLoc[0],
        isToggleChangedLoc[1],
        false,
        isToggleChangedLoc[3],
        isToggleChangedLoc[4],
        isToggleChangedLoc[5],
      ])
  }

  const handlePolyChange = (value) => {
    dispatch(variantFiltersActions.setPolyphen(value))
  }

  const handlePolyToggle = () => {
    isToggleChangedPatho[3] && dispatch(variantFiltersActions.togglePolyphen())
    isToggleChangedPatho[3] &&
      setIsToggleChangedPatho([
        isToggleChangedLoc[0],
        isToggleChangedLoc[1],
        isToggleChangedLoc[2],
        false,
        isToggleChangedLoc[4],
        isToggleChangedLoc[5],
      ])
  }

  const handleSiftChange = (value) => {
    dispatch(variantFiltersActions.setSift(value))
  }

  const handleSiftToggle = () => {
    isToggleChangedPatho[4] && dispatch(variantFiltersActions.toggleSift())
    isToggleChangedPatho[4] &&
      setIsToggleChangedPatho([
        isToggleChangedLoc[0],
        isToggleChangedLoc[1],
        isToggleChangedLoc[2],
        isToggleChangedLoc[3],
        false,
        isToggleChangedLoc[5],
      ])
  }

  // acmg filter states
  const [acmgEvent, setAcmgEvent] = useState([])

  // acmg filter functions
  const handleAcmgEventChange = () => {
    dispatch(variantFiltersActions.setAcmg([...acmgEvent]))
  }

  // clinvar filter states
  const [clinvarEvent, setClinvarEvent] = useState([])

  // clinvar filter functions
  const handleClinvarEventChange = () => {
    dispatch(variantFiltersActions.setClinVar([...clinvarEvent]))
  }
  // gene set filter states
  const [geneSets, setGeneSets] = useState([])

  const handleGeneSetChange = () => {
    const sets = geneSets.map((item) => item.value)
    if (sets.length > 0) dispatch(variantFiltersActions.setPredefined({ predefinedFilter: sets }))
  }

  // phenotype/omim filter states
  const [phenotypes, setPhenotypes] = useState([])

  const handlePhenotypeChange = () => {
    const phenos = phenotypes.map((item) => item.key)
    if (phenos.length > 0) dispatch(variantFiltersActions.setPhenotype({ diseaseFilter: phenos }))
  }

  // phenotype/omim filter states
  const [hpoFilter, setHpoFilter] = useState([])

  const handleHpoFilterChange = async () => {
    if (hpoFilter.length === 0) return
    const rawGeneNames = await Promise.all(hpoFilter.map((item) => hpoIdtoGeneName(item.value)))
    if (rawGeneNames.length === 0) return
    dispatch(variantFiltersActions.setHpo({ hpoFilter: [...new Set(rawGeneNames.flat())] }))
  }

  // ad-ab filters
  const [dpFilter, setDpFilter] = useState('')
  const [abFilter, setAbFilter] = useState('')
  const [dpFilterToggle, setDpFilterToggle] = useState(false)
  const [abFilterToggle, setAbFilterToggle] = useState(false)

  const handleDpFilterChange = (value) => {
    dispatch(variantFiltersActions.setDp(value))
    if (dpFilterToggle) dispatch(variantFiltersActions.toggleDp())
  }

  const handleAbFilterChange = (value) => {
    dispatch(variantFiltersActions.setAb(value))
    if (abFilterToggle) dispatch(variantFiltersActions.toggleAb())
  }

  // gene set filter states
  const [savedFilters, setSavedFilters] = useState({ value: state, name: '', label: '' })

  const isFilterDefault = (key, value) => {
    const zeroOne = [
      'af',
      'cadd',
      'dann',
      'metalR',
      'polyphen',
      'sift',
      'gnomAdFrequency',
      'exAcAf',
      'turkishVariomeAf',
      'oneKgFrequency',
    ]
    const zeroThousand = ['fsFilter', 'dp', 'ab']
    const zero150Thousand = ['qualFilter']
    const emptyArr = ['impacts', 'acmg', 'clinVar', 'gt', 'pvs1']
    const anyNone = ['inDbSnp', 'scenarioFilter', 'filter']
    const emptyStrArr = ['phenotypes', 'geneNames']

    if (zeroOne.includes(key)) return value.toString() === '0,1'
    else if (zeroThousand.includes(key)) return value.toString() === '0,1000'
    else if (zero150Thousand.includes(key)) return value.toString() === '0,150000'
    else if (emptyArr.includes(key)) return value.length === 0
    else if (anyNone.includes(key)) return value.toUpperCase() === 'ANY' || value === 'NONE'
    else if (emptyStrArr.includes(key)) return value.toString() === ''
    else if (!value) return true
    return false
  }

  const handleSavedFilterChange = () => {
    if (!savedFilters?.value) return
    // i apologize to the gods of programming for this
    // but it works
    // and i'm not sure how to do this better
    // so i'm just going to leave it
    Object.keys(savedFilters.value).forEach((key) => {
      if (isFilterDefault(key, savedFilters.value[key].value)) return
      if (savedFilters.value[key].value && savedFilters.value[key].value.toString() !== '0,1') {
        if (key === 'af') {
          dispatch(variantFiltersActions.setAf(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleAf())
        } else if (key === 'cadd') {
          dispatch(variantFiltersActions.setCadd(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleCadd())
        } else if (key === 'dann') {
          dispatch(variantFiltersActions.setDann(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleDann())
        } else if (key === 'metalR') {
          dispatch(variantFiltersActions.setMetalR(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleMetalR())
        } else if (key === 'chromFilter') {
          dispatch(variantFiltersActions.setChrom(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleChrom())
        } else if (key === 'fsFilter') {
          dispatch(variantFiltersActions.setFs(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleFs())
        } else if (key === 'gnomAdFrequency') {
          dispatch(variantFiltersActions.setGnomAdFrequency(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleGnomAdFrequency())
        } else if (key === 'exAcAf') {
          dispatch(variantFiltersActions.setExAcAf(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleExAcAf())
        } else if (key === 'turkishVariomeAf') {
          dispatch(variantFiltersActions.setTurkishVariomeAf(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleTurkishVariomeAf())
        } else if (key === 'impacts') {
          dispatch(variantFiltersActions.setImpacts(savedFilters.value[key].value))
        } else if (key === 'inDbsnp') {
          dispatch(variantFiltersActions.setInDbSnp(savedFilters.value[key].value))
        } else if (key === 'oneKgFrequency') {
          dispatch(variantFiltersActions.setOneKgFrequency(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleOneKgFrequency())
        } else if (key === 'polyphen') {
          dispatch(variantFiltersActions.setPolyphen(savedFilters.value[key].value))
          dispatch(variantFiltersActions.togglePolyphen())
        } else if (key === 'posFilter') {
          dispatch(variantFiltersActions.setPos(savedFilters.value[key].value))
          dispatch(variantFiltersActions.togglePos())
        } else if (key === 'qualFilter') {
          dispatch(variantFiltersActions.setQual(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleQual())
        } else if (key === 'scenarioFilter') {
          dispatch(variantFiltersActions.setScenario(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleScenario())
        } else if (key === 'sift') {
          dispatch(variantFiltersActions.setSift(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleSift())
        } else if (key === 'kh1') {
          dispatch(variantFiltersActions.setKh1(savedFilters.value[key].value))
        } else if (key === 'multiposFilter') {
          dispatch(variantFiltersActions.setMultipos(savedFilters.value[key].value))
        } else if (key === 'kh2') {
          dispatch(variantFiltersActions.setKh2(savedFilters.value[key].value))
        } else if (key === 'kh3') {
          dispatch(variantFiltersActions.setKh3(savedFilters.value[key].value))
        } else if (key === 'filter') {
          dispatch(variantFiltersActions.setFilter(savedFilters.value[key].value))
        } else if (key === 'gt') {
          dispatch(variantFiltersActions.setGt(savedFilters.value[key].value))
        } else if (key === 'geneNames') {
          dispatch(variantFiltersActions.setGeneNames(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleGeneNames())
        } else if (key === 'hpoFilter') {
          dispatch(variantFiltersActions.setHpo(savedFilters.value[key].value))
        } else if (key === 'predefinedFilter') {
          dispatch(variantFiltersActions.setPredefined(savedFilters.value[key].value))
        } else if (key === 'nonBenignFilter') {
          dispatch(variantFiltersActions.setNonBenign(savedFilters.value[key].value))
        } else if (key === 'acmg') {
          dispatch(variantFiltersActions.setAcmg(savedFilters.value[key].value))
        } else if (key === 'clinVar') {
          dispatch(variantFiltersActions.setClinVar(savedFilters.value[key].value))
        } else if (key === 'phenotypes') {
          dispatch(variantFiltersActions.setPhenotype(savedFilters.value[key].value))
        } else if (key === 'dp') {
          dispatch(variantFiltersActions.setDp(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleDp())
        } else if (key === 'ab') {
          dispatch(variantFiltersActions.setAb(savedFilters.value[key].value))
          dispatch(variantFiltersActions.toggleAb())
        }
      }
    })
  }
  // apply filter button handler
  const handleApplyFilter = () => {
    handleChromToggle()
    handlePosToggle()
    handleGeneNamesToggle()

    handleChromChange(chromEvent)
    handlePosChange(posEvent)
    handleGeneNamesChange()

    // genotype filter dispatch
    handleGtFilterChange(gtFilterEvent)

    // frequency filter dispatch
    handleAfToggle()
    handleOneKgToggle()
    handleGnomAdFrequencyToggle()
    handleExAcAfToggle()
    handleTurkishVariomeAfToggle()

    handleInDbsnpChange(inDbsnpEvent)
    handleAfChange(afEvent)
    handleOneKgChange(oneKgFrequencyEvent)
    handleGnomAdFrequencyChange(gnomAdFrequencyEvent)
    handleExAcAfChange(exAcAfEvent)
    handleTurkishVariomeAfChange(turkishVariomeAfEvent)

    // quality filter dispatch
    handleQualFToggle()
    handleFsFToggle()

    handleFilterFChange(filterFEvent)
    handleQualFChange(qualFEvent)
    handleFsFChange(fsFEvent)

    // impact filter dispatch
    handleImpactEventChange()

    // pathogenicity filter dispatch
    handleCaddToggle()
    handleDannToggle()
    handleMetalRToggle()
    handlePolyToggle()
    handleSiftToggle()

    handleCaddChange(caddEvent)
    handleDannChange(dannEvent)
    handleMetalRChange(metalREvent)
    handlePolyChange(polyEvent)
    handleSiftChange(siftEvent)

    // acmg filter dispatch
    handleAcmgEventChange()
    handleClinvarEventChange()

    // gene set filter dispatch
    handleGeneSetChange()
    handlePhenotypeChange()
    handleHpoFilterChange()

    // ad-ab filter dispatch
    handleDpFilterChange(dpFilter)
    handleAbFilterChange(abFilter)
    handleSavedFilterChange()
    props.onClose()
  }

  const handleSaveFilter = () => {
    return state
  }

  // const handleResetFilter = () => {
  //
  // }

  const renderPanelItem = ({ title, component, tooltip }) => {
    return (
      <FiltersAccordion
        title={title}
        tooltip={tooltip}
        // expanded={openedItems.has(title)}
        // onChange={() => handleChange(title)}
      >
        {component}
      </FiltersAccordion>
    )
  }

  return (
    <>
      <CardContent className={classes.cardContent}>
        {renderPanelItem({
          title: 'Location',
          component: (
            <LocationFilters
              chromEvent={chromEvent}
              setChromEvent={setChromEvent}
              posEvent={posEvent}
              setPosEvent={setPosEvent}
              geneNamesInputText={geneNamesInputText}
              setGeneNamesInputText={setGeneNamesInputText}
              toggleSelectorChrom={toggleSelectorChrom}
              setToggleSelectorChrom={setToggleSelectorChrom}
              toggleSelectorPos={toggleSelectorPos}
              setToggleSelectorPos={setToggleSelectorPos}
              toggleSelectorGene={toggleSelectorGene}
              setToggleSelectorGene={setToggleSelectorGene}
              isToggleChangedLoc={isToggleChangedLoc}
              setIsToggleChangedLoc={setIsToggleChangedLoc}
            />
          ),
        })}
        {renderPanelItem({
          title: 'Genotype',
          component: <GenotypeFilter gtFilterEvent={gtFilterEvent} setGtFilterEvent={setGtFilterEvent} />,
        })}
        {/* {renderPanelItem({ title: 'Scenario', component: <ScenarioFilters /> })} */}
        {renderPanelItem({
          title: 'Frequency',
          component: (
            <FrequencyFilters
              afEvent={afEvent}
              setAfEvent={setAfEvent}
              oneKgFrequencyEvent={oneKgFrequencyEvent}
              setOneKgFrequencyEvent={setOneKgFrequencyEvent}
              gnomAdFrequencyEvent={gnomAdFrequencyEvent}
              setGnomAdFrequencyEvent={setGnomAdFrequencyEvent}
              exAcAfEvent={exAcAfEvent}
              setExAcAfEvent={setExAcAfEvent}
              turkishVariomeAfEvent={turkishVariomeAfEvent}
              setTurkishVariomeAfEvent={setTurkishVariomeAfEvent}
              inDbsnpEvent={inDbsnpEvent}
              setInDbsnpEvent={setInDbsnpEvent}
              toggleSelectorAf={toggleSelectorAf}
              setToggleSelectorAf={setToggleSelectorAf}
              toggleSelectorOneKgFrequency={toggleSelectorOneKgFrequency}
              setToggleSelectorOneKgFrequency={setToggleSelectorOneKgFrequency}
              toggleSelectorGnomAdFrequency={toggleSelectorGnomAdFrequency}
              setToggleSelectorGnomAdFrequency={setToggleSelectorGnomAdFrequency}
              toggleSelectorExAcAf={toggleSelectorExAcAf}
              setToggleSelectorExAcAf={setToggleSelectorExAcAf}
              toggleSelectorTurkishVariomeAf={toggleSelectorTurkishVariomeAf}
              setToggleSelectorTurkishVariomeAf={setToggleSelectorTurkishVariomeAf}
              isToggleChangedFrequency={isToggleChangedFrequency}
              setIsToggleChangedFrequency={setIsToggleChangedFrequency}
            />
          ),
        })}
        {renderPanelItem({
          title: 'Quality',
          component: (
            <QualityFilters
              qualFEvent={qualFEvent}
              setQuallFEvent={setQuallFEvent}
              fsFEvent={fsFEvent}
              setFsFEvent={setFsFEvent}
              filterFEvent={filterFEvent}
              setfilterFEvent={setfilterFEvent}
              toggleSelectorQualF={toggleSelectorQualF}
              setToggleSelectorQualF={setToggleSelectorQualF}
              toggleSelectorFsF={toggleSelectorFsF}
              setToggleSelectorFsF={setToggleSelectorFsF}
              isToggleChangedQuality={isToggleChangedQuality}
              setIsToggleChangedQuality={setIsToggleChangedQuality}
            />
          ),
        })}
        {renderPanelItem({
          title: 'Impact',
          component: <ImpactFilters impactEvent={impactEvent} setImpactEvent={setImpactEvent} filterType="impact" />,
        })}
        {renderPanelItem({
          title: 'Pathogenicity',
          component: (
            <PathogenicityFilters
              caddEvent={caddEvent}
              setCaddEvent={setCaddEvent}
              dannEvent={dannEvent}
              setDannEvent={setDannEvent}
              metalREvent={metalREvent}
              setMetalREvent={setMetalREvent}
              polyEvent={polyEvent}
              setPolyEvent={setPolyEvent}
              siftEvent={siftEvent}
              setSiftEvent={setSiftEvent}
              setClinvarEvent={setClinvarEvent}
              toggleSelectorCadd={toggleSelectorCadd}
              setToggleSelectorCadd={setToggleSelectorCadd}
              toggleSelectorDann={toggleSelectorDann}
              setToggleSelectorDann={setToggleSelectorDann}
              toggleSelectorMetalR={toggleSelectorMetalR}
              setToggleSelectorMetalR={setToggleSelectorMetalR}
              toggleSelectorPoly={toggleSelectorPoly}
              setToggleSelectorPoly={setToggleSelectorPoly}
              toggleSelectorSift={toggleSelectorSift}
              setToggleSelectorSift={setToggleSelectorSift}
              isToggleChangedPatho={isToggleChangedPatho}
              setIsToggleChangedPatho={setIsToggleChangedPatho}
            />
          ),
        })}
        {renderPanelItem({
          title: 'ACMG',
          component: <ImpactFilters impactEvent={acmgEvent} setImpactEvent={setAcmgEvent} filterType="acmg" />,
          tooltip:
            'Li, Quan, and Kai Wang. "InterVar: clinical interpretation of genetic variants by the 2015 ACMG-AMP guidelines." The American Journal of Human Genetics 100.2 (2017): 267-280.',
        })}
        {renderPanelItem({
          title: 'ClinVar',
          component: <ImpactFilters impactEvent={clinvarEvent} setImpactEvent={setClinvarEvent} filterType="clinvar" />,
          tooltip:
            'Landrum, Melissa J., et al. "ClinVar: improvements to accessing data." Nucleic acids research 48.D1 (2020): D835-D844.',
        })}
        {renderPanelItem({
          title: 'Panels',
          component: <GeneSetFilters geneSets={geneSets} setGeneSets={setGeneSets} />,
        })}
        {renderPanelItem({
          title: 'Diseases',
          component: <PhenotypeFilter phenotypes={phenotypes} setPhenotypes={setPhenotypes} />,
          tooltip:
            'Pletscher-Frankild, Sune, et al. "DISEASES: Text mining and data integration of disease–gene associations." Methods 74 (2015): 83-89.',
        })}
        {renderPanelItem({
          title: 'HPO',
          component: <HPOFilter hpoFilter={hpoFilter} setHpoFilter={setHpoFilter} />,
          tooltip:
            'Köhler, Sebastian, et al. "The human phenotype ontology in 2017." Nucleic acids research 45.D1 (2017): D865-D876.',
        })}
        {renderPanelItem({
          title: 'Read Details',
          component: (
            <ReadFilters
              dpFilter={dpFilter}
              setDpFilter={setDpFilter}
              dpFilterToggle={dpFilterToggle}
              setDpFilterToggle={setDpFilterToggle}
              abFilter={abFilter}
              setAbFilter={setAbFilter}
              abFilterToggle={abFilterToggle}
              setAbFilterToggle={setAbFilterToggle}
            />
          ),
        })}
        {renderPanelItem({
          title: 'Saved Filters',
          component: (
            <SavedFilters chosenFilter={savedFilters} setChosenFilter={setSavedFilters} handleSave={handleSaveFilter} />
          ),
        })}
      </CardContent>
      <Box mt={2} display="flex" justifyContent="flex-end" alignItems="flex-end">
        {/* <Button variant='contained' color='error' sx={{ marginRight: '0.5rem' }} onClick={() => handleResetFilter()}>
            Reset Filters
          </Button> */}
        <Button variant="contained" onClick={handleApplyFilter}>
          Apply Filters
        </Button>
      </Box>
    </>
  )
}

export default FiltersPanelMenu
