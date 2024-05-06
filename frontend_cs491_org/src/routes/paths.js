// ----------------------------------------------------------------------

import NewVariantDashboard from "src/views/NewVariantDashboard"

function path(root, sublink) {
  return `${root}${sublink}`
}

const ROOTS_AUTH = '/'
const ROOTS_DASHBOARD = '/priovar'
const ROOTS_SAMPLE = '/sample'
export const ROOTS_PrioVar = 'http://localhost:8080'
export const ROOTS_Flask = 'http://localhost:5001'
// ----------------------------------------------------------------------

export const PATH_PrioVar = {
  root: ROOTS_PrioVar,
  clinician: {
    root: path(ROOTS_PrioVar, '/clinician'),
    get: path(ROOTS_PrioVar, '/:clinicianId'),
    add: path(ROOTS_PrioVar, '/add'),
    login: path(ROOTS_PrioVar, '/login'),
    changepassword: path(ROOTS_PrioVar, '/changePassword'),
    allPatients: path(ROOTS_PrioVar, '/allPatients/:clinicianId'),
  },
  healthCenter: {
    root: path(ROOTS_PrioVar, '/medicalCenter'),
    get: path(ROOTS_PrioVar, '/:medicalCenterId'),
    add: path(ROOTS_PrioVar, '/add'),
    login: path(ROOTS_PrioVar, '/login'),
    changepassword: path(ROOTS_PrioVar, '/changePassword'),
    addsubscription: path(ROOTS_PrioVar, "/addSubscription/:medicalCenterId/:subscriptionId")
  },
  admin: {
    root: path(ROOTS_PrioVar, '/admin'),
    add: path(ROOTS_PrioVar, '/add'),
    login: path(ROOTS_PrioVar, '/login'),
    changepassword: path(ROOTS_PrioVar, '/changePassword'),
  },
  patient: {
    root: path(ROOTS_PrioVar, '/patient'),
    get: path(ROOTS_PrioVar, '/:patientId'),
    getByDisease: path(ROOTS_PrioVar, '/byDisease/:diseaseId'),
    getByMedicalCenter: path(ROOTS_PrioVar, '/byMedicalCenter/:medicalCenterId'),
    getByClinician: path(ROOTS_PrioVar, '/byClinician/:clinicianId'),
    add: path(ROOTS_PrioVar, '/add'),
    login: path(ROOTS_PrioVar, '/login'),
    changepassword: path(ROOTS_PrioVar, '/changePassword'),
  },
}

export const PATH_AUTH = {
  root: ROOTS_AUTH,
  login: path(ROOTS_AUTH, ''),
  loginUnprotected: path(ROOTS_AUTH, '/login-unprotected'),
  loginHealthCenter: path(ROOTS_AUTH, '/login-health-center'),
  loginHealthCenterUnprotected: path(ROOTS_AUTH, '/login-health-center-unprotected'),
  loginAdmin: path(ROOTS_AUTH, '/login-admin'),
  loginAdminUnprotected: path(ROOTS_AUTH, '/login-admin-unprotected'),
  register: path(ROOTS_AUTH, '/register'),
  registerUnprotected: path(ROOTS_AUTH, '/register-unprotected'),
  resetPassword: path(ROOTS_AUTH, '/reset-password'),
  verify: path(ROOTS_AUTH, '/verify'),
}

export const PATH_PAGE = {
  comingSoon: '/coming-soon',
  maintenance: '/maintenance',
  pricing: '/pricing',
  payment: '/payment',
  about: '/about-us',
  contact: '/contact-us',
  faqs: '/faqs',
  page404: '/404',
  page500: '/500',
  components: '/components',
}

export const PATH_SAMPLE = {
  sample: '/sample',
}

export const PATH_DASHBOARD = {
  root: ROOTS_DASHBOARD,
  general: {
    app: path(ROOTS_DASHBOARD, '/app'),
    ecommerce: path(ROOTS_DASHBOARD, '/ecommerce'),
    analytics: path(ROOTS_DASHBOARD, '/analytics'),
    banking: path(ROOTS_DASHBOARD, '/banking'),
    booking: path(ROOTS_DASHBOARD, '/booking'),
    files: path(ROOTS_DASHBOARD, '/files'),
    samples: path(ROOTS_DASHBOARD, '/samples'),
    variantDashboard: path(ROOTS_DASHBOARD, '/sample/:fileId/:sampleName'),
    variants: path(ROOTS_DASHBOARD, '/variants/:fileId/:sampleName?'),
    variantDetails: path(ROOTS_DASHBOARD, '/variants/:fileId/:sampleName/:chrom/:pos'),
    myPatients: path(ROOTS_DASHBOARD, '/clinician/:healthCenterId/patients'),
    clinicPatients: path(ROOTS_DASHBOARD, '/clinics/:healthCenterId/patients'),
    requestedPatients: path(ROOTS_DASHBOARD, '/clinics/:healthCenterId/requestedPatients'),
    customQuery: path(ROOTS_DASHBOARD, '/customquery'),
    subscriptionPlans: path(ROOTS_DASHBOARD, '/subscriptionPlans'),
    similarPatients: path(ROOTS_DASHBOARD, '/similarPatients'),
    addClinician: path(ROOTS_DASHBOARD, '/addClinician'),
    patientDetails: path(ROOTS_DASHBOARD, '/patientDetails/:patientId/:fileId'),
    patientDetailsConst: path(ROOTS_DASHBOARD, '/patientDetailsConst/:patientId'),
    aiSupport: path(ROOTS_DASHBOARD, '/aiSupport'),
    informationRetrieval: path(ROOTS_DASHBOARD, '/informationRetrieval'),
    NewVariantDashboard: path(ROOTS_DASHBOARD, '/sample/:fileName')
  },
  mail: {
    root: path(ROOTS_DASHBOARD, '/mail'),
    all: path(ROOTS_DASHBOARD, '/mail/all'),
  },
  chat: {
    root: path(ROOTS_DASHBOARD, '/chat'),
    new: path(ROOTS_DASHBOARD, '/chat/new'),
    conversation: path(ROOTS_DASHBOARD, '/chat/:conversationKey'),
  },
  calendar: path(ROOTS_DASHBOARD, '/calendar'),
  kanban: path(ROOTS_DASHBOARD, '/kanban'),
  user: {
    root: path(ROOTS_DASHBOARD, '/user'),
    profile: path(ROOTS_DASHBOARD, '/user/profile'),
    cards: path(ROOTS_DASHBOARD, '/user/cards'),
    list: path(ROOTS_DASHBOARD, '/user/list'),
    newUser: path(ROOTS_DASHBOARD, '/user/new'),
    editById: path(ROOTS_DASHBOARD, `/user/reece-chung/edit`),
    account: path(ROOTS_DASHBOARD, '/user/account'),
  },
  eCommerce: {
    root: path(ROOTS_DASHBOARD, '/e-commerce'),
    shop: path(ROOTS_DASHBOARD, '/e-commerce/shop'),
    product: path(ROOTS_DASHBOARD, '/e-commerce/product/:name'),
    productById: path(ROOTS_DASHBOARD, '/e-commerce/product/nike-air-force-1-ndestrukt'),
    list: path(ROOTS_DASHBOARD, '/e-commerce/list'),
    newProduct: path(ROOTS_DASHBOARD, '/e-commerce/product/new'),
    editById: path(ROOTS_DASHBOARD, '/e-commerce/product/nike-blazer-low-77-vintage/edit'),
    checkout: path(ROOTS_DASHBOARD, '/e-commerce/checkout'),
    invoice: path(ROOTS_DASHBOARD, '/e-commerce/invoice'),
  },
  blog: {
    root: path(ROOTS_DASHBOARD, '/blog'),
    posts: path(ROOTS_DASHBOARD, '/blog/posts'),
    post: path(ROOTS_DASHBOARD, '/blog/post/:title'),
    postById: path(ROOTS_DASHBOARD, '/blog/post/apply-these-7-secret-techniques-to-improve-event'),
    newPost: path(ROOTS_DASHBOARD, '/blog/new-post'),
  },
}

export const PATH_DOCS = 'https://docs-minimals.vercel.app/introduction'
