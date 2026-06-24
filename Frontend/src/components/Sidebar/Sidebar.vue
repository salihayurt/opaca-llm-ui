<template>
    <div id="sidebar-base" class="d-flex">
        <!-- sidebar selection -->
        <div id="sidebar-menu"
             class="d-flex flex-column justify-content-start align-items-center gap-2">

            <!-- Always Visible: Info -->
            <i @click="SidebarManager.toggleView('info')"
               class="fa fa-circle-info sidebar-menu-item"
               :title="Localizer.get('sidebar_info')"
               v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('info')}" />

            <!-- Always Visible: Chats -->
            <i @click="SidebarManager.toggleView('chats')"
               class="fa fa-message sidebar-menu-item"
               :title="Localizer.get('sidebar_chats')"
               v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('chats')}" />

            <!-- Always Visible: Prompts -->
            <i @click="SidebarManager.toggleView('questions')"
               class="fa fa-book sidebar-menu-item"
               :title="Localizer.get('sidebar_questions')"
               v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('questions')}" />

            <!-- Collapse/Extend Button -->
            <i @click="toggleSidebar()"
               @mouseenter="sidebarToggleHovered = true"
               @mouseleave="sidebarToggleHovered = false"
               class="fa sidebar-menu-toggle"
               :class="getSidebarToggleIcon()"
               :title="getSidebarToggleTooltip()" />

            <!-- Expanded-only tools -->
            <div v-if="!sidebarCollapsed"
                 class="d-flex flex-column align-items-center gap-2"
                 style="min-height: 0; overflow: hidden;">
                <i @click="SidebarManager.toggleView('files')"
                   class="fa fa-file sidebar-menu-item"
                   :title="Localizer.get('sidebar_files')"
                   v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('files')}" />

                <i @click="SidebarManager.toggleView('agents')"
                   class="fa fa-users sidebar-menu-item"
                   :title="Localizer.get('sidebar_agents')"
                   v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('agents')}"/>

                <i @click="SidebarManager.toggleView('extensions')"
                   class="fa fa-puzzle-piece sidebar-menu-item"
                   :title="Localizer.get('sidebar_extensions')"
                   v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('extensions')}"/>

                <i @click="SidebarManager.toggleView('mcp')"
                   class="fa fa-server sidebar-menu-item"
                   :title="Localizer.get('sidebar_mcp')"
                   v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('mcp')}"/>

                <i @click="SidebarManager.toggleView('config')"
                   class="fa fa-cog sidebar-menu-item"
                   :title="Localizer.get('sidebar_config')"
                   v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('config')}"/>

                <i @click="SidebarManager.toggleView('debug')"
                   class="fa fa-bug sidebar-menu-item"
                   :title="Localizer.get('sidebar_logs')"
                   v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('debug')}"/>
            </div>

            <!-- spacer -->
            <div class="flex-grow-1" />

            <!-- Always Visible: FAQ -->
            <i @click="SidebarManager.toggleView('faq')"
               class="fa fa-question-circle sidebar-menu-item"
               :title="Localizer.get('sidebar_faq')"
               v-bind:class="{'sidebar-menu-item-select': SidebarManager.isViewSelected('faq')}"/>

            <!-- Always Visible: User Profile -->
            <div class="sidebar-account-wrapper">
                <div class="sidebar-menu-item sidebar-avatar-wrapper"
                     @click.stop="toggleProfileMenu()"
                     :class="{'sidebar-menu-item-select': accountMenuOpen}"
                     :title="Localizer.get('sidebar_account')">
                     <img v-if="isAuthenticated && user?.picture"
                          :src="user.picture"
                          class="sidebar-avatar"
                          alt="User avatar"/>
                     <i v-else class="fa fa-user"/>
                </div>

                <div v-if="accountMenuOpen"
                     class="sidebar-account-menu"
                     @click.stop>
                    <button type="button"
                            class="sidebar-account-menu-button"
                            @click="handleProfileAuthClick()">
                        <i :class="['fa', this.isAuthenticated ? 'fa-right-from-bracket' : 'fa-right-to-bracket']"/>
                        <span>{{ this.isAuthenticated ? Localizer.get('account_logout') : Localizer.get('account_login') }}</span>
                    </button>
                    <button type="button"
                            class="sidebar-account-menu-button"
                            @click="handleProfileSettingsClick()">
                        <i class="fa fa-gear"/>
                        <span>{{ Localizer.get('settings_menu') }}</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- sidebar content -->
        <div v-show="SidebarManager.isSidebarOpen()">
            <aside id="sidebar-content"
                   class="d-flex flex-column position-relative">

                <!-- platform information -->
                <SidebarInfo
                    v-show="SidebarManager.isViewSelected('info')"
                    :is-platform-connected="connected"
                    :sidebar-view="SidebarManager.getSelectedView()"
                    ref="info"
                />

                <!-- chats -->
                <SidebarChats
                    v-show="SidebarManager.isViewSelected('chats')"
                    :selected-chat-id="this.selectedChatId"
                    :is-finished="this.isFinished"
                    :chats="this.chats"
                    @select-chat="chatId => this.$emit('select-chat', chatId)"
                    @delete-chat="chatId => this.$emit('delete-chat', chatId)"
                    @rename-chat="(chatId, newName) => this.$emit('rename-chat', chatId, newName)"
                    @new-chat="() => this.$emit('new-chat')"
                    @goto-search-result="(chatId, messageId) => this.$emit('goto-search-result', chatId, messageId)"
                    @delete-all-chats="() => this.$emit('delete-all-chats')"
                    @update-chats="this.updateChats"
                    ref="chats"
                />

                <!-- uploaded files -->
                <SidebarFiles
                    :selectedChatId="this.selectedChatId"
                    :chats="this.chats"
                    v-show="SidebarManager.isViewSelected('files')"
                    @delete-file="fileId => this.$emit('delete-file', fileId)"
                    @view-file="$emit('view-file', $event)"
                    @rename-file="(fileId, newName) => this.$emit('rename-file', fileId, newName)"
                    @update-chats="this.updateChats"
                    ref="files"
                />

                <!-- sample questions -->
                <SidebarQuestions
                    v-show="SidebarManager.isViewSelected('questions')"
                    @select-question="question => this.$emit('select-question', question)"
                    ref="questions"
                />

                <!-- agents/actions overview -->
                <SidebarAgents
                    v-show="SidebarManager.isViewSelected('agents')"
                    :is-platform-connected="connected"
                    ref="agents"
                />

                <!-- UI extensions -->
                <SidebarExtensions
                    v-show="SidebarManager.isViewSelected('extensions')"
                    :is-platform-connected="connected"
                    ref="extensions"
                />

                <!-- MCP servers -->
                <SidebarMcp
                    v-show="SidebarManager.isViewSelected('mcp')"
                    :is-platform-connected="connected"
                    ref="mcp"
                />

                <!-- method config -->
                <SidebarConfig
                    v-show="SidebarManager.isViewSelected('config')"
                    ref="config"
                />

                <!-- debug console -->
                <SidebarDebug
                    v-show="SidebarManager.isViewSelected('debug')"
                    :selected-chat-id="this.selectedChatId"
                    ref="debug"
                />

                <!-- Help/FAQ -->
                <SidebarFaq
                    v-show="SidebarManager.isViewSelected('faq')"
                    ref="faq"
                />

                <div v-show="!isMobile" class="resizer" id="resizer" />
            </aside>
        </div>
    </div>
</template>

<script>
import conf from '../../../config.js'
import { useDevice } from "../../useIsMobile.js";
import SidebarManager from "../../SidebarManager.js";
import { useAuthentication } from "../../useAuthentication.ts";
import Localizer from "../../Localizer.js";
import SidebarQuestions from './SidebarQuestions.vue';
import SidebarAgents from "./SidebarAgents.vue";
import SidebarExtensions from './SidebarExtensions.vue';
import SidebarConfig from "./SidebarConfig.vue";
import SidebarInfo from "./SidebarInfo.vue";
import SidebarDebug from "./SidebarDebug.vue";
import SidebarFaq from "./SidebarFaq.vue";
import SidebarChats from "./SidebarChats.vue";
import SidebarFiles from "./SidebarFiles.vue";
import SidebarMcp from "./SidebarMcp.vue";
import backendClient from "../../utils.js";

export default {
    name: 'Sidebar',
    components: {
        SidebarMcp,
        SidebarFiles,
        SidebarChats,
        SidebarFaq,
        SidebarDebug,
        SidebarInfo,
        SidebarConfig,
        SidebarAgents,
        SidebarExtensions,
        SidebarQuestions,
    },
    props: {
        connected: Boolean,
        selectedChatId: String,
        isFinished: Boolean,
    },
    emits: [
        'select-question',
        'select-chat',
        'delete-chat',
        'rename-chat',
        'new-chat',
        'delete-file',
        'view-file',
        'rename-file',
        'goto-search-result',
        'delete-all-chats',
    ],
    setup() {
        const { isMobile } = useDevice();
        const { loginWithPopup, logout, user, isAuthenticated } = useAuthentication();
        return { SidebarManager, Localizer, isMobile, loginWithPopup, logout, user, isAuthenticated };
    },
    data() {
        return {
            sidebarCollapsed: conf.sidebarCollapsed,
            sidebarToggleHovered: false,
            chats: [],
            accountMenuOpen: false,
        };
    },
    methods: {
        toggleProfileMenu() {
            this.accountMenuOpen = !this.accountMenuOpen;
        },

        closeProfileMenu() {
            this.accountMenuOpen = false;
        },

        async handleProfileAuthClick() {
            if (this.isAuthenticated) {
                await this.logout({ logoutParams: { returnTo: window.location.origin } });
            } else {
                try {
                    await this.loginWithPopup({authorizationParams: {screen_hint: 'signup'}})
                } catch (error) {
                    // Only show an error in the console, if the popup was not closed
                    if (error.error === 'cancelled') return;
                    console.error("Auth0 login failed: ", error)
                }
            }
            // Update the sidebar information to reflect the new user state
            await this.updateSidebarUserInfo();
            // MCP Servers need to be updated separately
            await this.$refs.mcp.updateMcp(this.connected);
        },

        async handleProfileSettingsClick() {
            console.log("Not implemented yet.")
        },

        toggleSidebar() {
            conf.sidebarCollapsed = !conf.sidebarCollapsed;
            this.sidebarCollapsed = conf.sidebarCollapsed; // needed for auto-update

            // Close view if it is now hidden
            const view = this.SidebarManager.getSelectedView();
            if (this.SidebarManager.viewNotInCollapsed(view)) {
                this.SidebarManager.close();
            }
        },

        getSidebarToggleIcon() {
            if (conf.sidebarCollapsed) return 'fa-angle-down';
            if (this.sidebarToggleHovered) return 'fa-angle-up';
            return 'fa-minus';
        },

        getSidebarToggleTooltip() {
            if (conf.sidebarCollapsed) return Localizer.get('sidebar_showAdvancedTools');
            return Localizer.get('sidebar_showStandardTools');
        },

        setupResizer() {
            const resizer = document.getElementById('resizer');
            const sidebar = document.getElementById('sidebar-content');

            resizer.addEventListener('mousedown', (e) => {
                SidebarManager.setResizing(true);
                document.body.style.cursor = 'ew-resize';
            });

            document.addEventListener('mousemove', (event) => {
                if (!SidebarManager.isResizing()) return;

                // Calculate the new width for the aside
                const newWidth = event.clientX - sidebar.getBoundingClientRect().left;

                if (newWidth > 200 && newWidth < 768) {
                    sidebar.style.width = `${newWidth}px`;
                }
            });

            document.addEventListener('mouseup', () => {
                SidebarManager.setResizing(false);
                document.body.style.cursor = 'default';
            });
        },

        async updateChats() {
            try {
                this.chats = await backendClient.chats();
            } catch (error) {
                console.error(error);
                this.chats = [];
            }
        },

        async updateSidebarUserInfo() {
            // All-in-One Place to update user data in the sidebar
            // Called by App.vue
            await this.updateChats();
            await this.$refs.files.updateFiles();
            await this.$refs.config.fetchMethodConfig();
            await this.$refs.questions.loadPrompts();
        }
    },
    mounted() {
        this.setupResizer();
        document.addEventListener('click', this.closeProfileMenu);

        if (this.isMobile) {
            SidebarManager.close()
        } else {
            SidebarManager.selectView(conf.selectedSidebar, conf.sidebarCollapsed);
        }
    },
    beforeUnmount() {
        document.removeEventListener('click', this.closeProfileMenu);
    },
}
</script>

<style>
/* used in the sub-components! */
.sidebar-title {
    display: flex;
    align-items: center;
    font-size: 150%;
    border-left: 5px solid var(--primary-color);
    padding-left: .5em;
    margin-bottom: .5em;
}
</style>

<style scoped>
#sidebar-base {
    background-color: var(--background-color);
}

/* sidebar content */
#sidebar-content {
    width: min(400px, 100vw - 3rem);
    height: calc(100vh - 50px - 1rem - 1rem); /* 100% - header - top margin - bottom margin */
    min-width: 150px;
    max-width: 768px;
    padding: .5rem;
    margin: 1rem 0 0 1rem;
    z-index: 999;
    background-color: var(--surface-color);
    border-radius: .5rem;
}

#sidebar-menu {
    background-color: var(--surface-color);
    border-right: 1px solid var(--border-color);
    padding: 0.5rem;
    margin: 1rem 0 0 1rem;
    transition: all 0.2s ease;
    border-radius: 0.5rem;
    height: calc(100vh - 50px - 1rem - 1rem); /* 100% - header - top margin - bottom margin */
}

.sidebar-menu-item {
    font-size: 1.25rem;
    cursor: pointer;
    width: 3rem;
    height: 3rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--bs-border-radius-lg);
    color: var(--text-secondary-color);
    transition: all 0.2s ease;
}

.sidebar-menu-item:hover {
    background-color: var(--background-color);
    color: var(--primary-color);
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}

.sidebar-menu-item-select {
    background-color: var(--primary-color) !important;
    color: white !important;
}

.sidebar-menu-item-select:hover {
    background-color: var(--secondary-color);
    color: white !important;
}

.sidebar-account-wrapper {
    position: relative;
}

.sidebar-account-menu {
    position: absolute;
    left: calc(100% + 0.5rem);
    bottom: 0;
    min-width: 10rem;
    padding: 0.35rem;
    background-color: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: var(--bs-border-radius);
    z-index: 1001;
}

.sidebar-account-menu-button {
    width: 100%;
    min-height: 2.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.65rem;
    border: 0;
    border-radius: var(--bs-border-radius-sm);
    background: transparent;
    color: var(--text-primary-color);
    text-align: left;
    white-space: nowrap;
}

.sidebar-account-menu-button:hover {
    background-color: var(--background-color);
    color: var(--primary-color);
}

.sidebar-account-menu-button i {
    width: 1rem;
    text-align: center;
}

.sidebar-menu-toggle {
    font-size: 1.25rem;
    cursor: pointer;
    width: 3rem;
    height: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary-color);
    transition: all 0.2s ease;
}

.sidebar-menu-toggle:hover {
    color: var(--primary-color);
}

.sidebar-avatar-wrapper {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}

.sidebar-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    object-fit: cover;
}

.resizer {
    width: 4px;
    cursor: ew-resize;
    height: calc(100vh - 50px - 1rem - 1rem); /* same as content */
    position: absolute;
    top: 0;
    right: 0;
    border-radius: var(--bs-border-radius-sm);
    background-color: var(--border-color);
    transition: background-color 0.2s ease;
}

.resizer:hover {
    background-color: var(--primary-color);
}

/* mobile design */
@media screen and (max-width: 768px) {
    .resizer {
        display: none;
    }

    #sidebar-menu {
        padding: 0.25rem;
        margin: 0;
        height: calc(100vh - 50px);
    }

    .sidebar-menu-item {
        font-size: 1rem;
        width: 2.5rem;
        height: 2.5rem;
    }

    .sidebar-menu-toggle {
        width: 2.5rem;
        height: 1rem;
        font-size: 1rem;
    }

    #sidebar-content {
        width: min(600px, 100vw - 3rem);
        height: calc(100vh - 50px);
        padding-left: 0;
        padding-right: 0;
        margin: 0;
    }

}

</style>
