<template>
<div class="container flex-grow-1 overflow-hidden overflow-y-auto">
    <div v-if="!isMobile" class="sidebar-title">
        {{ Localizer.get('sidebar_promptMacros') }}
    </div>

    <div v-if="isLoading">
        <i class="fa fa-circle-notch fa-spin me-1" />
        {{ Localizer.get('promptMacros_loading') }}
    </div>

    <div v-if="errorMessage" class="text-danger mb-2">
        {{ errorMessage }}
    </div>

    <div v-if="!isLoading && !errorMessage && promptMacros.length === 0">
        {{ Localizer.get('promptMacros_missing') }}
    </div>

    <AppAccordion
        v-if="!isLoading && promptMacros.length > 0"
        id="prompt-macros-accordion"
        class="text-start"
        :items="sortedPromptMacros"
        :get-key="macro => macro.id"
    >
        <template #header="{ item: macro }">
            <i class="fa fa-cubes-stacked me-3"
               :class="{ 'macro-disabled': !macro.enabled }" />
            <strong class="macro-name"
                    :class="{ 'macro-disabled': !macro.enabled }">
                {{ macro.name }}
            </strong>

            <span class="macro-actions">
                <i class="fa fa-lg macro-action"
                   :class="macro.enabled ? 'fa-toggle-on' : 'fa-toggle-off'"
                   @click.stop="toggleMacro(macro.id)"
                   :title="Localizer.get(macro.enabled ? 'promptMacros_disable' : 'promptMacros_enable')" />
                <i class="fa fa-remove macro-action"
                   @click.stop="deleteMacro(macro)"
                   :title="Localizer.get('promptMacros_delete')" />
            </span>
        </template>

        <template #body="{ item: macro }">
            <div class="macro-body"
                 :class="{ 'macro-disabled': !macro.enabled }">
                <div class="macro-section">
                    <strong>{{ Localizer.get('promptMacros_whenToUse') }}</strong>
                    <p>{{ macro.when_to_use }}</p>
                </div>
                <div class="macro-section">
                    <strong>{{ Localizer.get('promptMacros_whatToDo') }}</strong>
                    <p class="macro-instructions">{{ macro.what_to_do }}</p>
                </div>
            </div>
        </template>
    </AppAccordion>

    <button type="button"
            class="btn btn-primary py-2 w-100"
            :disabled="isSaving"
            @click.stop="addPromptMacro">
        <i class="fa fa-plus me-2" />
        {{ Localizer.get('promptMacros_add') }}
    </button>

    <InputDialogue ref="input" />
</div>
</template>

<script>
import AppAccordion from "../AppAccordion.vue";
import InputDialogue from "../InputDialogue.vue";
import Localizer from "../../Localizer.js";
import backendClient from "../../utils.js";
import { useDevice } from "../../useIsMobile.js";

export default {
    name: "SidebarPromptMacros",
    components: { AppAccordion, InputDialogue },
    props: {
        sidebarView: String,
    },
    setup() {
        const { isMobile } = useDevice();
        return { Localizer, isMobile };
    },
    data() {
        return {
            promptMacros: [],
            isLoading: false,
            isSaving: false,
            errorMessage: "",
        };
    },
    methods: {
        async loadPromptMacros() {
            this.isLoading = true;
            this.errorMessage = "";
            try {
                this.promptMacros = await backendClient.getPromptMacros();
            } catch (error) {
                console.error("Failed to load prompt macros", error);
                this.errorMessage = Localizer.get("promptMacros_loadFailed");
            } finally {
                this.isLoading = false;
            }
        },

        async addPromptMacro() {
            await this.$refs.input.showDialogue(
                Localizer.get("promptMacros_add"),
                null,
                null,
                {
                    name: {
                        type: "text",
                        label: Localizer.get("promptMacros_name"),
                    },
                    when_to_use: {
                        type: "textarea",
                        label: Localizer.get("promptMacros_whenToUse"),
                        rows: 4,
                    },
                    what_to_do: {
                        type: "textarea",
                        label: Localizer.get("promptMacros_whatToDo"),
                        rows: 7,
                    },
                },
                async values => {
                    const macroData = {
                        name: values.name.trim(),
                        when_to_use: values.when_to_use.trim(),
                        what_to_do: values.what_to_do.trim(),
                    };
                    this.isSaving = true;
                    this.errorMessage = "";
                    try {
                        const createdMacro = await backendClient.savePromptMacro(macroData);
                        this.promptMacros.push(createdMacro);
                    } catch (error) {
                        console.error("Failed to save prompt macro", error);
                        this.errorMessage = Localizer.get("promptMacros_saveFailed");
                        throw new Error(Localizer.get("promptMacros_saveFailed"));
                    } finally {
                        this.isSaving = false;
                    }
                },
            );
        },

        async toggleMacro(macroId) {
            if (this.isSaving) return;
            const macroIndex = this.promptMacros.findIndex(macro => macro.id === macroId);
            if (macroIndex < 0) return;

            const macro = this.promptMacros[macroIndex];
            const enabled = !macro.enabled;
            this.isSaving = true;
            this.errorMessage = "";
            try {
                const updatedMacro = await backendClient.setPromptMacroEnabled(macro, enabled);
                this.promptMacros[macroIndex] = updatedMacro;
            } catch (error) {
                console.error("Failed to update prompt macro", error);
                this.errorMessage = Localizer.get("promptMacros_saveFailed");
            } finally {
                this.isSaving = false;
            }
        },

        async deleteMacro(macro) {
            if (this.isSaving) return;
            await this.$refs.input.showDialogue(
                Localizer.get("promptMacros_delete"),
                Localizer.get("promptMacros_delete_confirm", macro.name),
                null,
                {},
                async () => {
                    this.isSaving = true;
                    this.errorMessage = "";
                    try {
                        await backendClient.deletePromptMacro(macro.id);
                        this.promptMacros = this.promptMacros.filter(item => item.id !== macro.id);
                    } catch (error) {
                        console.error("Failed to delete prompt macro", error);
                        this.errorMessage = Localizer.get("promptMacros_deleteFailed");
                        throw new Error(Localizer.get("promptMacros_deleteFailed"));
                    } finally {
                        this.isSaving = false;
                    }
                },
            );
        },
    },
    computed: {
        sortedPromptMacros() {
            return [...this.promptMacros]
                .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
        },
    },
    watch: {
        sidebarView(newView) {
            if (newView === "promptMacros") {
                this.loadPromptMacros();
            }
        },
    },
};
</script>

<style scoped>
.macro-name {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.macro-actions {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
}

.macro-action {
    flex: 0 0 auto;
    width: 2rem;
    height: 2rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    cursor: pointer;
    margin-right: 0 !important;
}

.macro-action:hover {
    background-color: var(--input-color);
}

.fa-toggle-on {
    color: var(--text-success-color);
}

.fa-toggle-off {
    color: var(--text-secondary-color);
}

.fa-remove:hover {
    color: var(--text-danger-color);
}

.macro-disabled {
    opacity: 0.5;
}

.macro-body {
    padding: 0.75rem;
}

.macro-section + .macro-section {
    margin-top: 1rem;
}

.macro-section p {
    margin: 0.25rem 0 0;
    overflow-wrap: anywhere;
}

.macro-instructions {
    white-space: pre-wrap;
}
</style>
