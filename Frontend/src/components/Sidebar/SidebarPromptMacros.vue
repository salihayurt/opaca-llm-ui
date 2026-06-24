<template>
<div class="container d-flex flex-column flex-grow-1 overflow-hidden">
    <InputDialogue ref="input" />

    <div v-if="!isMobile" class="sidebar-title">
        {{ Localizer.get('sidebar_promptMacros') }}
    </div>

    <div class="flex-grow-1 overflow-y-auto">
        <div v-if="isLoading" class="text-secondary p-3">
            <i class="fa fa-circle-notch fa-spin me-1" />
            {{ Localizer.get('promptMacros_loading') }}
        </div>

        <div v-else-if="errorMessage" class="text-danger p-3">
            {{ errorMessage }}
        </div>

        <div v-else-if="promptMacros.length === 0" class="text-secondary p-4">
            {{ Localizer.get('promptMacros_missing') }}
        </div>

        <div v-else id="prompt-macros-accordion" class="accordion text-start">
            <div v-for="(macro, index) in promptMacros"
                 :key="macro.id"
                 class="accordion-item"
                 :class="{ 'macro-disabled': !macro.enabled }">
                <h2 class="accordion-header m-0">
                    <button class="accordion-button collapsed"
                            type="button"
                            data-bs-toggle="collapse"
                            :data-bs-target="`#prompt-macro-${index}`"
                            aria-expanded="false"
                            :aria-controls="`prompt-macro-${index}`">
                        <i class="fa fa-cubes-stacked me-3" />
                        <strong class="macro-name">{{ macro.name }}</strong>

                        <span class="macro-actions ms-auto">
                            <i class="fa fa-lg macro-action"
                               :class="macro.enabled ? 'fa-toggle-on' : 'fa-toggle-off'"
                               @click.stop="toggleMacro(macro)"
                               :title="Localizer.get(macro.enabled ? 'promptMacros_disable' : 'promptMacros_enable')" />
                            <i class="fa fa-remove macro-action"
                               @click.stop="deleteMacro(macro)"
                               :title="Localizer.get('promptMacros_delete')" />
                        </span>
                    </button>
                </h2>

                <div :id="`prompt-macro-${index}`"
                     class="accordion-collapse collapse"
                     data-bs-parent="#prompt-macros-accordion">
                    <div class="accordion-body">
                        <div class="macro-section">
                            <strong>{{ Localizer.get('promptMacros_whenToUse') }}</strong>
                            <p>{{ macro.when_to_use }}</p>
                        </div>
                        <div class="macro-section">
                            <strong>{{ Localizer.get('promptMacros_whatToDo') }}</strong>
                            <p class="macro-instructions">{{ macro.what_to_do }}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <button type="button"
            class="btn btn-primary py-2 w-100 mt-3"
            :disabled="isSaving"
            @click="addPromptMacro">
        <i class="fa fa-plus" />
        {{ Localizer.get('promptMacros_add') }}
    </button>
</div>
</template>

<script>
import InputDialogue from "../InputDialogue.vue";
import Localizer from "../../Localizer.js";
import backendClient from "../../utils.js";
import { useDevice } from "../../useIsMobile.js";

export default {
    name: "SidebarPromptMacros",
    components: { InputDialogue },
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
                    const macro = {
                        id: this.createMacroId(values.name),
                        name: values.name.trim(),
                        when_to_use: values.when_to_use.trim(),
                        what_to_do: values.what_to_do.trim(),
                        enabled: true,
                    };
                    this.isSaving = true;
                    this.errorMessage = "";
                    try {
                        await backendClient.savePromptMacro(macro);
                        this.promptMacros.push(macro);
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

        createMacroId(name) {
            const slug = name.toLowerCase()
                .trim()
                .replace(/[^a-z0-9]+/g, "_")
                .replace(/^_+|_+$/g, "") || "prompt_macro";
            return `${slug}_${Date.now().toString(36)}`;
        },

        async toggleMacro(macro) {
            if (this.isSaving) return;
            const enabled = !macro.enabled;
            this.isSaving = true;
            this.errorMessage = "";
            try {
                await backendClient.setPromptMacroEnabled(macro, enabled);
                macro.enabled = enabled;
            } catch (error) {
                console.error("Failed to update prompt macro", error);
                this.errorMessage = Localizer.get("promptMacros_saveFailed");
            } finally {
                this.isSaving = false;
            }
        },

        async deleteMacro(macro) {
            if (this.isSaving || !confirm(Localizer.get("promptMacros_delete_confirm", macro.name))) return;
            this.isSaving = true;
            this.errorMessage = "";
            try {
                await backendClient.deletePromptMacro(macro.id);
                this.promptMacros = this.promptMacros.filter(item => item.id !== macro.id);
            } catch (error) {
                console.error("Failed to delete prompt macro", error);
                this.errorMessage = Localizer.get("promptMacros_deleteFailed");
            } finally {
                this.isSaving = false;
            }
        },
    },
    mounted() {
        this.loadPromptMacros();
    },
};
</script>

<style scoped>
.macro-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.macro-actions {
    display: inline-flex;
    align-items: center;
    padding-right: 1.5rem;
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

.macro-disabled .macro-name,
.macro-disabled .fa-cubes-stacked,
.macro-disabled .accordion-body {
    opacity: 0.5;
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
