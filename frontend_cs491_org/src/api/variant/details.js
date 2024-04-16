import { maxBy, reduce, zipObj } from 'ramda'
import { useQuery } from 'react-query'
import { API_BASE_URL, CONSEQUENCE_TO_IMPACT_MAP } from '../../constants'
import axios from '../../utils/axios'

const hydratePandasDataFrame = (df) => {
  if (!df) {
    return []
  }

  const { data, columns } = df

  return data.map(zipObj(columns))
}
const fixDiseaseData = (disease) => {
  Object.entries(disease).forEach(([geneName, geneDiseaseData]) => {
    geneDiseaseData.textmining = hydratePandasDataFrame(geneDiseaseData.textmining)
    geneDiseaseData.experiments = hydratePandasDataFrame(geneDiseaseData.experiments)
    geneDiseaseData.knowledge = hydratePandasDataFrame(geneDiseaseData.knowledge)
  })
}
const maxByImpact = maxBy((impact) => ['MODIFIER', 'LOW', 'MODERATE', 'HIGH'].indexOf(impact))
const getMaxImpact = reduce(maxByImpact, 'MODIFIER')
const hydrateImpactData = (transcripts) => {
  transcripts.forEach((t) => {
    const impacts = t.Consequences.map((c) => CONSEQUENCE_TO_IMPACT_MAP[c])
    t.Impact = getMaxImpact(impacts)
  })
}
const fetchVariantDetails = async (fileId, sampleName, chrom, pos) => {
  const { data } = await axios.get(`${API_BASE_URL}/variant-details/${fileId}/${sampleName}/${chrom}/${pos}`)

  const { variant, disease, omim, matchmaking } = data

  fixDiseaseData(disease)
  hydrateImpactData(variant.Transcripts)

  return { variant, disease, omim, matchmaking }
}
export const useVariantDetails = ({ fileId, sampleName, chrom, pos }) => {
  return useQuery(['variant-details', fileId, sampleName, chrom, pos], () =>
    fetchVariantDetails(fileId, sampleName, chrom, pos),
  )
}
