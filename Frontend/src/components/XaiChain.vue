<template>
<div class="xai-chain">
    <div v-if="loading" class="p-2">
        <i class="fa fa-spin fa-circle-o-notch me-1" /> {{ Localizer.get('xai_loading') }}
    </div>

    <div v-else-if="error" class="p-2 text-danger">{{ error }}</div>

    <div v-else-if="chain">
        <!-- summary line: everything here is counted, not inferred -->
        <div class="xai-summary small mb-2">
            {{ chain.nodes.length }} {{ Localizer.get('xai_calls') }}
            <span v-if="chain.failures > 0" class="text-danger">
                · {{ chain.failures }} {{ Localizer.get('xai_failed') }}
            </span>
            <span v-if="chain.untraceable > 0" class="xai-warn">
                · {{ chain.untraceable }} {{ Localizer.get('xai_untraceable') }}
            </span>
            <span class="text-muted"> · {{ chain.execution_time.toFixed(1) }}s</span>
            <span :class="confidenceClass">
                · {{ Localizer.get('xai_confidence_' + chain.confidence) }}
            </span>
        </div>

        <!-- What lowered it. A bare level asks to be trusted; the list says
             what to check, which is what lets someone disagree with it. -->
        <ul v-if="chain.signals.length > 0" class="xai-signals small mb-2">
            <li v-for="signal in chain.signals" :key="signal.id">{{ signal.detail }}</li>
        </ul>

        <div v-if="chain.nodes.length === 0" class="small text-muted p-2">
            {{ Localizer.get('xai_noCalls') }}
        </div>

        <svg v-else :viewBox="`0 0 ${width} ${height}`"
             :style="{ width: '100%', height: height + 'px' }"
             class="xai-svg">
            <!-- arrows are drawn first so nodes sit on top of them -->
            <defs>
                <marker id="xai-arrow" viewBox="0 0 8 8" refX="7" refY="4"
                        markerWidth="6" markerHeight="6" orient="auto">
                    <path d="M0,0 L8,4 L0,8 z" class="xai-arrowhead" />
                </marker>
            </defs>

            <g v-for="(link, i) in layout.links" :key="'l' + i">
                <path :d="link.path" class="xai-link" marker-end="url(#xai-arrow)" />
                <text :x="link.labelX" :y="link.labelY" class="xai-link-label"
                      text-anchor="middle">{{ link.param }}</text>
            </g>

            <g v-for="node in layout.nodes" :key="node.id"
               class="xai-node" :class="{ selected: selectedId === node.id }"
               @click.stop="select(node.id)">
                <rect :x="node.x" :y="node.y" :width="nodeWidth" :height="nodeHeight"
                      rx="6" class="xai-node-box"
                      :class="{ failed: node.success === false, pending: node.success === null }" />
                <text :x="node.x + nodeWidth / 2" :y="node.y + 20"
                      text-anchor="middle" class="xai-node-label">{{ truncate(node.action) }}</text>
                <text :x="node.x + nodeWidth / 2" :y="node.y + 36"
                      text-anchor="middle" class="xai-node-agent">{{ truncate(node.agent, 18) }}</text>
                <text v-if="node.success === false" :x="node.x + nodeWidth - 10" :y="node.y + 16"
                      text-anchor="end" class="xai-node-status">✕</text>
            </g>
        </svg>

        <!-- detail for the selected node -->
        <div v-if="selected" class="xai-detail small mt-2 p-2 rounded-2">
            <div class="fw-bold mb-1">{{ selected.name }}</div>
            <div v-if="selected.error" class="text-danger mb-1">{{ selected.error }}</div>
            <!-- The model's own reason, captured with the call. Marked as a
                 quote because it is a self-report, unlike the traced values. -->
            <blockquote v-if="selected.rationale" class="xai-rationale mb-1">
                {{ selected.rationale }}
                <div v-if="selected.considered" class="xai-considered">
                    {{ Localizer.get('xai_considered') }} {{ selected.considered }}
                </div>
            </blockquote>
            <div v-if="selected.sources.length === 0" class="text-muted">
                {{ Localizer.get('xai_noParams') }}
            </div>
            <div v-for="source in selected.sources" :key="source.param" class="xai-param">
                <code>{{ source.param }}</code>
                <span class="xai-value">{{ truncate(source.value, 30) }}</span>
                <span :class="sourceClass(source)">{{ describe(source) }}</span>
            </div>
        </div>
        <div v-else class="small text-muted mt-2">{{ Localizer.get('xai_selectHint') }}</div>

        <!-- The only part that costs an LLM call, so it only happens on a click. -->
        <div class="mt-2">
            <button v-if="!summary && !summarising" class="btn btn-sm btn-outline-secondary"
                    @click="explain">
                {{ Localizer.get('xai_explain') }}
            </button>
            <span v-if="summarising" class="small text-muted">
                <i class="fa fa-spin fa-circle-o-notch me-1" />{{ Localizer.get('xai_explaining') }}
            </span>

            <div v-if="summary" class="xai-detail small p-2 rounded-2">
                <div class="fw-bold mb-1">{{ summary.headline }}</div>
                <ol v-if="summary.steps.length" class="mb-1 ps-3">
                    <li v-for="(step, i) in summary.steps" :key="i">
                        <span class="fw-semibold">{{ step.title }}</span> — {{ step.detail }}
                    </li>
                </ol>
                <ul v-if="summary.caveats.length" class="xai-signals mb-1">
                    <li v-for="(caveat, i) in summary.caveats" :key="'c' + i">{{ caveat }}</li>
                </ul>
                <!-- Said plainly rather than hidden: a written account and a
                     counted one are different kinds of claim. -->
                <div class="text-muted xai-provenance-note">
                    {{ Localizer.get(summary.generated ? 'xai_written' : 'xai_counted') }}
                </div>
                <!-- Named rather than swallowed: a panel that quietly degrades
                     gives no way to tell a broken explainer from a chain there
                     was nothing to say about. -->
                <div v-if="summary.error" class="xai-src-weak xai-provenance-note">
                    {{ Localizer.get('xai_explainFailed') }} {{ summary.error }}
                </div>
            </div>
        </div>
    </div>
</div>
</template>

<script>
import backendClient from "../utils.js";
import Localizer from "../Localizer.js";

/**
 * The call chain of one response: which tools ran, and which argument values
 * came from which earlier result.
 *
 * Everything drawn is computed in the backend from the recorded trace, so the
 * graph cannot show a step that did not happen. An argument the model supplied
 * itself has no arrow pointing at it, which is the same finding the approval
 * prompt states in words.
 *
 * The layout is hand-rolled rather than pulled from a graph library: chains
 * run to a handful of nodes, one column per iteration places them adequately,
 * and a dependency would cost more than it saves.
 */
export default {
    name: 'XaiChain',
    props: {
        chatId: String,
        responseId: String,
    },
    setup() {
        return { Localizer };
    },
    data() {
        return {
            chain: null,
            loading: false,
            error: null,
            selectedId: null,
            summary: null,
            summarising: false,
            nodeWidth: 150,
            nodeHeight: 46,
            columnGap: 62,
            rowGap: 20,
        };
    },
    computed: {
        selected() {
            return this.chain?.nodes.find(n => n.id === this.selectedId) ?? null;
        },

        /**
         * Nodes in columns, a node one column right of whatever feeds it.
         *
         * Iteration would be the obvious grouping and is the wrong one: calls
         * that depend on each other can happen in the same round, and then
         * every node lands in one column with the arrows folded on top of it.
         * Depth in the data-flow graph is what the arrows actually draw, so
         * the layout follows that. Calls that depend on nothing share the
         * first column, whether they ran first or not.
         */
        columns() {
            const nodes = this.chain?.nodes ?? [];
            const links = this.chain?.links ?? [];
            const depth = {};
            nodes.forEach(n => { depth[n.id] = 0; });

            // Chains are short, so settling this by repetition is cheaper than
            // a topological sort, and it cannot loop: a value can only come
            // from a call that already ran.
            for (let pass = 0; pass < nodes.length; pass++) {
                let moved = false;
                links.forEach(link => {
                    const wanted = (depth[link.from_id] ?? 0) + 1;
                    if (wanted > (depth[link.to_id] ?? 0)) {
                        depth[link.to_id] = wanted;
                        moved = true;
                    }
                });
                if (!moved) break;
            }

            const groups = new Map();
            nodes.forEach(node => {
                const column = depth[node.id] ?? 0;
                if (!groups.has(column)) groups.set(column, []);
                groups.get(column).push(node);
            });
            return [...groups.keys()].sort((a, b) => a - b).map(k => groups.get(k));
        },

        width() {
            return Math.max(1, this.columns.length) * (this.nodeWidth + this.columnGap);
        },

        height() {
            const tallest = Math.max(1, ...this.columns.map(c => c.length));
            return tallest * (this.nodeHeight + this.rowGap) + this.rowGap;
        },

        layout() {
            const positions = {};
            const nodes = [];
            this.columns.forEach((column, columnIndex) => {
                column.forEach((node, rowIndex) => {
                    const x = columnIndex * (this.nodeWidth + this.columnGap) + this.columnGap / 2;
                    const y = rowIndex * (this.nodeHeight + this.rowGap) + this.rowGap;
                    positions[node.id] = { x, y };
                    nodes.push({ ...node, x, y });
                });
            });

            const links = (this.chain?.links ?? []).map(link => {
                const from = positions[link.from_id];
                const to = positions[link.to_id];
                if (!from || !to) return null;
                const x1 = from.x + this.nodeWidth;
                const y1 = from.y + this.nodeHeight / 2;
                const x2 = to.x;
                const y2 = to.y + this.nodeHeight / 2;
                const mid = (x1 + x2) / 2;
                return {
                    // A curve rather than a straight line: two arrows between
                    // the same pair of columns would otherwise overlap exactly.
                    path: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`,
                    labelX: mid,
                    labelY: (y1 + y2) / 2 - 5,
                    param: link.param,
                };
            }).filter(Boolean);

            return { nodes, links };
        },
    },
    watch: {
        responseId: 'load',
    },
    mounted() {
        this.load();
    },
    methods: {
        async load() {
            // Returning quietly here once cost an afternoon: a prop name that
            // did not bind left chatId undefined, and the panel simply showed
            // nothing with no request in the network tab and no console error.
            if (!this.chatId || !this.responseId) {
                this.error = Localizer.get('xai_loadFailed');
                console.warn('XaiChain: missing chatId or responseId',
                             {chatId: this.chatId, responseId: this.responseId});
                return;
            }
            this.loading = true;
            this.error = null;
            try {
                this.chain = await backendClient.responseChain(this.chatId, this.responseId);
                this.summary = null;
            } catch (e) {
                this.error = Localizer.get('xai_loadFailed');
            } finally {
                this.loading = false;
            }
        },

        async explain() {
            this.summarising = true;
            try {
                this.summary = await backendClient.explainResponse(this.chatId, this.responseId);
            } catch (e) {
                this.summary = null;
                this.error = Localizer.get('xai_loadFailed');
            } finally {
                this.summarising = false;
            }
        },

        select(id) {
            this.selectedId = this.selectedId === id ? null : id;
        },

        truncate(text, max = 20) {
            const value = String(text ?? '');
            return value.length > max ? value.slice(0, max - 1) + '…' : value;
        },

        describe(source) {
            const action = (source.origin ?? '').split(' ')[0].split('--').pop();
            switch (source.source) {
                case 'user_query': return Localizer.get('confirm_fromYourRequest');
                case 'tool_result': return `${Localizer.get('confirm_fromResult')} ${action}`;
                case 'prior_argument': return `${Localizer.get('confirm_reused')} ${action}`;
                default:
                    return source.kind === 'weak'
                        ? Localizer.get('confirm_notChecked')
                        : Localizer.get('confirm_notTraceable');
            }
        },

        confidenceClass() {
            return {
                high: 'xai-src-ok',
                medium: 'xai-warn-amber',
                low: 'xai-warn',
            }[this.chain?.confidence] ?? '';
        },

        sourceClass(source) {
            if (source.kind === 'weak') return 'xai-src-weak';
            return source.source === 'unmatched' ? 'xai-src-warn' : 'xai-src-ok';
        },
    },
}
</script>

<style scoped>
.xai-svg {
    max-height: 320px;
}

.xai-node {
    cursor: pointer;
}

.xai-node-box {
    fill: var(--bs-body-bg, #fff);
    stroke: var(--bs-border-color, #ccc);
    stroke-width: 1.5;
}

.xai-node-box.failed {
    stroke: var(--bs-danger, #dc3545);
}

.xai-node-box.pending {
    stroke-dasharray: 4 3;
}

.xai-node.selected .xai-node-box {
    stroke: var(--bs-primary, #0d6efd);
    stroke-width: 2.5;
}

.xai-node-label {
    font-size: 12px;
    font-weight: 600;
    fill: var(--bs-body-color, #212529);
}

.xai-node-agent,
.xai-link-label {
    font-size: 10px;
    fill: var(--bs-secondary-color, #6c757d);
}

.xai-node-status {
    font-size: 12px;
    fill: var(--bs-danger, #dc3545);
}

.xai-link {
    fill: none;
    stroke: var(--bs-border-color, #ccc);
    stroke-width: 1.5;
}

.xai-arrowhead {
    fill: var(--bs-border-color, #ccc);
}

.xai-detail {
    background-color: var(--bs-tertiary-bg, #f8f9fa);
}

.xai-param {
    display: flex;
    gap: .5rem;
    align-items: baseline;
    flex-wrap: wrap;
    padding: 2px 0;
}

.xai-value {
    font-family: monospace;
    opacity: .8;
}

.xai-rationale {
    border-left: 3px solid var(--bs-border-color, #ccc);
    padding-left: .6rem;
    margin: 0;
    font-style: italic;
    opacity: .9;
}

.xai-considered {
    font-style: normal;
    opacity: .75;
    margin-top: .25rem;
}

.xai-src-ok { color: var(--bs-success, #198754); }
.xai-src-warn { color: var(--bs-danger, #dc3545); font-weight: 600; }
.xai-src-weak { color: var(--bs-secondary-color, #6c757d); }
.xai-warn { color: var(--bs-danger, #dc3545); }
.xai-warn-amber { color: var(--bs-warning-text-emphasis, #997404); }

.xai-signals {
    list-style: none;
    padding-left: 0;
    margin-bottom: .5rem;
    color: var(--bs-secondary-color, #6c757d);
}

.xai-signals li::before {
    content: "· ";
}

.xai-provenance-note {
    font-size: .85em;
    margin-top: .35rem;
}
</style>