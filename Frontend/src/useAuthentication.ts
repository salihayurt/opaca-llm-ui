import {computed} from "vue";
import {useAuth0} from "@auth0/auth0-vue";
import conf from '../config.js';

/* Helper wrapper to turn authentication on and off */

export function useAuthentication() {
    if (!conf.authEnabled) {
        return {
            isAuthenticated: computed(() => false),
            user: computed(() => null),
            loginWithPopup: async () => {},
            logout: async () => {},
            getAccessTokenSilently: async () => null
        };
    }

    return useAuth0();
}