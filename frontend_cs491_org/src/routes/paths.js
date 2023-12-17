// ----------------------------------------------------------------------

function path(root, sublink) {
  return `${root}${sublink}`
}

const ROOTS_AUTH = '/auth'
const ROOTS_DASHBOARD = '/libra'
const ROOTS_SAMPLE = '/sample'
export const ROOTS_PRIOVAR = 'http://localhost:8080'
// ----------------------------------------------------------------------

export const PATH_PRIOVAR = {
  root: ROOTS_PRIOVAR,
  clinician: {
    root: path(ROOTS_PRIOVAR, '/clinician'),
    get: path(ROOTS_PRIOVAR, '/:clinicianId'),
    add: path(ROOTS_PRIOVAR, '/add'),
    login: path(ROOTS_PRIOVAR, '/login'),
    changepassword: path(ROOTS_PRIOVAR, '/changePassword'),
  },
  healthCenter: {
    root: path(ROOTS_PRIOVAR, '/medicalCenter'),
    get: path(ROOTS_PRIOVAR, '/:medicalCenterId'),
    add: path(ROOTS_PRIOVAR, '/add'),
    login: path(ROOTS_PRIOVAR, '/login'),
    changepassword: path(ROOTS_PRIOVAR, '/changePassword'),
    addsubscription: path(ROOTS_PRIOVAR, "/addSubscription/:medicalCenterId/:subscriptionId")
  },
  admin: {
    root: path(ROOTS_PRIOVAR, '/admin'),
    add: path(ROOTS_PRIOVAR, '/add'),
    login: path(ROOTS_PRIOVAR, '/login'),
    changepassword: path(ROOTS_PRIOVAR, '/changePassword'),
  },
}

export const PATH_AUTH = {
  root: ROOTS_AUTH,
  login: path(ROOTS_AUTH, '/login'),
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
    customQuery: path(ROOTS_DASHBOARD, '/customquery')
    //add my patients, clinics patients, subscription plans
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
