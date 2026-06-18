<template>
    <div :id="id" class="accordion app-accordion" :class="`app-accordion--${variant}`">
        <div
            v-for="(item, index) in items"
            :key="getItemKey(item, index)"
            class="accordion-item"
        >
            <h2 class="accordion-header" :id="getHeaderId(index)">
                <button
                    class="accordion-button"
                    :class="[buttonClass, { collapsed: !isOpen(item, index) }]"
                    type="button"
                    :aria-expanded="isOpen(item, index) ? 'true' : 'false'"
                    :aria-controls="getBodyId(index)"
                    @click="toggleItem(item, index)"
                >
                    <slot name="header" :item="item" :index="index" />
                </button>
            </h2>

            <div
                :id="getBodyId(index)"
                v-show="isOpen(item, index)"
                class="accordion-collapse"
                :aria-labelledby="getHeaderId(index)"
            >
                <div class="accordion-body">
                    <slot name="body" :item="item" :index="index" />
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'AppAccordion',
    props: {
        id: { type: String, required: true },
        items: { type: Array, default: () => [] },
        getKey: { type: Function, default: null },
        modelValue: { type: [String, Number, Object], default: undefined },
        variant: { type: String, default: 'default' },
        buttonClass: { type: [String, Array, Object], default: '' },
    },
    emits: ['update:modelValue'],
    data() {
        return {
            internalValue: null,
        };
    },
    computed: {
        activeValue() {
            return this.modelValue === undefined ? this.internalValue : this.modelValue;
        },
    },
    methods: {
        getItemKey(item, index) {
            return this.getKey ? this.getKey(item, index) : item?.id ?? index;
        },

        getHeaderId(index) {
            return `${this.id}-header-${index}`;
        },

        getBodyId(index) {
            return `${this.id}-body-${index}`;
        },

        isOpen(item, index) {
            const key = this.getItemKey(item, index);
            return this.activeValue === key;
        },

        toggleItem(item, index) {
            const key = this.getItemKey(item, index);
            const nextValue = this.isOpen(item, index) ? null : key;

            if (this.modelValue === undefined) this.internalValue = nextValue;

            this.$emit('update:modelValue', nextValue);
        },
    },
};
</script>

<style scoped>
.accordion-item {
    border-radius: var(--bs-border-radius);
    margin-bottom: 0.5rem;
    border: 1px solid var(--border-color);
    overflow: hidden;
    background-color: var(--surface-color);
}

.accordion-button {
    border-radius: var(--bs-border-radius);
    padding: 1rem;
    font-weight: 500;
    transition: all 0.2s ease;
    background-color: var(--background-color);
    color: var(--text-primary-color);
    border: 1px solid var(--border-color);
}

.accordion-button :deep(i) {
    margin-right: 0.75rem;
}

.accordion-button:not(.collapsed) {
    color: var(--button-primary-color);
    background-color: var(--primary-color);
    box-shadow: none;
}

.accordion-button:hover {
    color: var(--button-primary-color);
    background-color: var(--secondary-color)
}

.accordion-button:focus {
    box-shadow: none;
    border-color: transparent;
}

.accordion-button::after {
    background-size: 1rem;
    width: 1rem;
    height: 1rem;
    transition: all 0.2s ease;
    filter: invert(var(--icon-invert-color));
}

.accordion-body {
    padding: 0;
    background-color: var(--background-color);
    color: var(--text-primary-color);
}

.accordion-collapse {
    background-color: var(--background-color);
    color: var(--text-primary-color);
}

.app-accordion--nested .accordion-item {
    border: none;
    border-radius: 0;
    margin-bottom: 0;
    background-color: transparent;
}

.app-accordion--nested .accordion-header {
    margin-bottom: 0;
}

.app-accordion--nested .accordion-button {
    border-left: none;
    border-right: none;
    border-radius: 0;
    padding: 0.8rem 1rem;
}

.app-accordion--nested .accordion-body {
    padding: 0;
}

.app-accordion--compact .accordion-item {
    min-width: min(300px, calc(100vw - 6rem));
    max-width: calc(100vw - 6rem);
    margin-bottom: 0;
}

.app-accordion--compact .accordion-header,
.app-accordion--compact .accordion-item,
.app-accordion--compact .accordion-button {
    border-radius: 0 !important;
}

.app-accordion--compact .accordion-button {
    padding: 0.5rem !important;
}

.app-accordion--compact .accordion-button :deep(.text-muted) {
    transition: color 0.2s ease;
}

.app-accordion--compact .accordion-button:hover :deep(.text-muted),
.app-accordion--compact .accordion-button:not(.collapsed) :deep(.text-muted) {
    color: var(--text-primary-color) !important;
}
</style>
