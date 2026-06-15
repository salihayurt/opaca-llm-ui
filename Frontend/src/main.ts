import { createApp } from 'vue'
import "bootstrap/dist/css/bootstrap.min.css"
import "@fortawesome/fontawesome-free/css/all.min.css"
import "primeicons/primeicons.css"

import PrimeVue from "primevue/config"
import Aura from "@primeuix/themes/aura"
import {createAuth0} from "@auth0/auth0-vue"

import './style.css'
import App from './App.vue'

const app = createApp(App)

app.use(PrimeVue, {
    theme: {
        preset: Aura,
    },
})

app.use(createAuth0({
    domain: import.meta.env.VITE_AUTH0_DOMAIN,
    clientId: import.meta.env.VITE_AUTH0_CLIENT_ID,
    authorizationParams: {
        redirect_uri: window.location.origin,
        audience: import.meta.env.VITE_AUTH0_AUDIENCE,
    },
    cacheLocation: "localstorage",
    useRefreshTokens: true,
}))

app.mount('#app')
