<template>
<div class="container flex-grow-1 overflow-hidden overflow-y-auto">
    <div class="sidebar-title">
        {{ Localizer.get('sidebar_chats') }}
        <i class="fa fa-plus ms-auto sidebar-title-action"
           @click="this.$emit('new-chat')"
           :title="Localizer.get('chats_new')" />
        <i v-if="showHeaderActions"
           class="fa fa-magnifying-glass sidebar-title-action sidebar-title-action-secondary"
           :class="{ disabled: this.chats.length === 0 }"
           :aria-disabled="this.chats.length === 0"
           @click="this.isSearching = !this.isSearching"
           :title="Localizer.get('chats_search')" />
        <i v-if="showHeaderActions"
           class="fa fa-trash sidebar-title-action sidebar-title-action-danger"
           :class="{ disabled: this.chats.length === 0 }"
           :aria-disabled="this.chats.length === 0"
           @click="onDeleteAllChats"
           :title="Localizer.get('chats_deleteAll')" />
        <i class="fa sidebar-title-action sidebar-title-action-secondary"
           :class="showHeaderActions ? 'fa-angle-right' : 'fa-ellipsis'"
           @click="toggleHeaderActions"
           :title="Localizer.get(showHeaderActions ? 'sidebar_lessActions' : 'sidebar_moreActions')" />
    </div>

    <SidebarChatSearch
        v-if="isSearching"
        @stop-search="this.isSearching = false"
        @goto-search-result="this.gotoSearchResult"
    />

    <!-- List all the chats -->
    <div v-for="chat in chats" :key="chat.chat_id">
        <SidebarChatItem
            :selected-chat-id="this.selectedChatId"
            :chat-id="chat.chat_id"
            :chat="chat"
            :has-missed-response="isChatMissed(chat.chat_id)"
            @select-chat="chatId => this.$emit('select-chat', chatId)"
            @delete-chat="chatId => this.$emit('delete-chat', chatId)"
            @rename-chat="(chatId, name) => this.$emit('rename-chat', chatId, name)"
        />
    </div>
</div>
</template>

<script>
import Localizer from "../../Localizer.js";
import {useDevice} from "../../useIsMobile.js";
import SidebarChatItem from "./SidebarChatItem.vue";
import SidebarChatSearch from "../SidebarChatSearch.vue";

export default {
    name: 'SidebarChats',
    components: {SidebarChatSearch, SidebarChatItem},
    props: {
        selectedChatId: String,
        chats: Array,
    },
    setup() {
        const {isMobile} = useDevice();
        return { Localizer, isMobile};
    },
    emits: [
        'select-chat',
        'delete-chat',
        'rename-chat',
        'new-chat',
        'goto-search-result',
        'delete-all-chats',
        'update-chats',
    ],
    data() {
        return {
            missedResponseChatIds: [],
            showHeaderActions: false,
            isSearching: false,
        };
    },
    methods: {
        toggleHeaderActions() {
            this.showHeaderActions = !this.showHeaderActions;
            if (!this.showHeaderActions) {
                this.isSearching = false;
            }
        },

        async updateChats() {
            this.$emit('update-chats');
        },

        markChatMissed(chatId) {
            if (!chatId || this.missedResponseChatIds.includes(chatId)) return;
            this.missedResponseChatIds = [...this.missedResponseChatIds, chatId];
        },

        clearChatMissed(chatId = null) {
            if (chatId) {
                this.missedResponseChatIds = this.missedResponseChatIds.filter(id => id !== chatId);
            } else {
                this.missedResponseChatIds = [];
            }
        },

        isChatMissed(chatId) {
            return this.missedResponseChatIds.includes(chatId);
        },

        gotoSearchResult(chatId, messageId) {
            this.isSearching = false;
            this.$emit('goto-search-result', chatId, messageId)
        },

        onDeleteAllChats() {
            if (confirm(Localizer.get('chats_deleteAll_confirm'))) {
                this.$emit('delete-all-chats');
            }
        },
    },
    mounted() {
        //this.updateChats(); // ... is called in this stage, but moved to App.mounted to fix concurrency issues
    },
}
</script>

<style scoped>
</style>
