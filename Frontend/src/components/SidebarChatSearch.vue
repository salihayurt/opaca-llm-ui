<template>
<div>
    <input v-model="searchText"
           ref="searchBar"
           type="search"
           class="form-control my-2"
           :placeholder="Localizer.get('chats_search')"
           @input="this.updateSearchResults"
           @keyup.enter="this.updateSearchResults"
           @keyup.esc="this.$emit('stop-search')" />

    <div v-if="Object.keys(this.searchResults).length > 0" class="search-result-list">
        <div v-for="(results, chatId) in searchResults" :key="chatId" class="mt-3">
            <div class="small" style="color: var(--secondary-color)">
                {{ results?.[0].chat_name }}
            </div>
            <div v-for="(result, index) in results"
                 :key="index"
                 class="search-result"
                 @click="this.gotoResult(result)">
                {{ result.excerpt }}
            </div>
        </div>
    </div>
</div>
</template>

<script>
import backendClient from "../utils.js";
import Localizer from "../Localizer.js";

export default {
    name: "SidebarChatSearch",
    data() {
        return {
            searchText: '',
            searchResults: {},
            isLoadingResults: false,
        };
    },
    setup() {
        return { Localizer };
    },
    emits: [
        'stop-search',
        'goto-search-result',
    ],
    methods: {
        async updateSearchResults() {
            if (this.searchText.length < 3) {
                this.searchResults = {};
                return;
            }
            try {
                this.isLoadingResults = true;
                this.searchResults = await backendClient.search(this.searchText);
            } catch (error) {
                console.error(error);
                this.searchResults = {};
            } finally {
                this.isLoadingResults = false;
            }
        },

        gotoResult(result) {
            this.$emit('goto-search-result', result.chat_id, result.message_id);
        },

        clear() {
            this.searchText = '';
            this.searchResults = {};
            this.isLoadingResults = false;
        },
    },
    mounted() {
        this.clear();
        this.$refs.searchBar.focus();
    },
}
</script>

<style scoped>
.search-result-list {
    max-height: 50vh;
    overflow: auto;
}

.search-result {
    padding: 0.5rem;
    margin-top: 0.25rem;
}

.search-result:hover {
    background-color: var(--surface-color);
    cursor: pointer;
}
</style>
