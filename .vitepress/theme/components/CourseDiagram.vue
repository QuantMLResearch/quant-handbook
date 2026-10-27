<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import Mermaid from 'vitepress-plugin-mermaid/Mermaid.vue'

// The markdown plugin passes class="mermaid". Do not forward it to the
// viewer: Mermaid's automatic scan must never parse the toolbar as a graph.
defineOptions({ inheritAttrs: false })
const props = defineProps<{ graph: string; id: string }>()
const viewport = ref<HTMLElement>()
const canvas = ref<HTMLElement>()
const scale = ref(1)
const fit = ref(false)
const ready = ref(false)
const scrollLeft = ref(0)
const maxScroll = ref(0)
const percentage = computed(() => Math.round(scale.value * 100))
let svg: SVGSVGElement | null = null
let naturalWidth = 0
let naturalHeight = 0
let mutation: MutationObserver | undefined
let resize: ResizeObserver | undefined
let disposed = false

function measureScroll() {
  const el = viewport.value
  if (!el) return
  maxScroll.value = Math.max(0, el.scrollWidth - el.clientWidth)
  scrollLeft.value = el.scrollLeft
}
function applySize() {
  if (!svg || !naturalWidth || !viewport.value) return
  if (fit.value) scale.value = Math.min(1, (viewport.value.clientWidth - 32) / naturalWidth)
  svg.style.width = `${naturalWidth * scale.value}px`
  svg.style.height = `${naturalHeight * scale.value}px`
  measureScroll()
}
function readDiagram() {
  const next = canvas.value?.querySelector<SVGSVGElement>(':scope > .mermaid > svg')
  if (!next || next === svg) return
  svg = next
  const box = svg.viewBox.baseVal
  naturalWidth = box.width
  naturalHeight = box.height
  ready.value = naturalWidth > 0 && naturalHeight > 0
  applySize()
}
function zoom(value: number) {
  fit.value = false
  scale.value = Math.max(.15, Math.min(2, value))
  applySize()
}
function fitDiagram() {
  fit.value = true
  applySize()
}
function pan(value: number) {
  if (!viewport.value) return
  viewport.value.scrollLeft = value
  measureScroll()
}
function step(direction: number) {
  pan(scrollLeft.value + direction * (viewport.value?.clientWidth ?? 300) * .75)
}
function onKey(event: KeyboardEvent) {
  if (event.target !== viewport.value) return
  if (event.key === 'ArrowLeft') step(-1)
  else if (event.key === 'ArrowRight') step(1)
  else if (event.key === 'Home') pan(0)
  else if (event.key === 'End') pan(maxScroll.value)
  else return
  event.preventDefault()
}
onMounted(async () => {
  await nextTick()
  if (disposed || !canvas.value || !viewport.value) return
  mutation = new MutationObserver(readDiagram)
  mutation.observe(canvas.value, { childList: true, subtree: true })
  resize = new ResizeObserver(() => { applySize(); measureScroll() })
  resize.observe(viewport.value)
  readDiagram()
})
onBeforeUnmount(() => { disposed = true; mutation?.disconnect(); resize?.disconnect() })
</script>

<template>
  <figure class="course-diagram" aria-label="课程图示">
    <div class="diagram-toolbar">
      <span class="diagram-label">课程图示</span>
      <div class="diagram-tools" role="group" aria-label="图示缩放">
        <button type="button" aria-label="缩小图示" :disabled="!ready || scale <= .15" @click="zoom(scale - .15)">−</button>
        <button type="button" class="diagram-scale" aria-label="恢复图示原始大小" :disabled="!ready" @click="zoom(1)">{{ percentage }}%</button>
        <button type="button" aria-label="放大图示" :disabled="!ready || scale >= 2" @click="zoom(scale + .15)">＋</button>
        <button type="button" class="diagram-fit" :aria-pressed="fit" :disabled="!ready" @click="fitDiagram">全图</button>
      </div>
    </div>
    <div ref="viewport" class="diagram-viewport" tabindex="0" role="region" aria-label="图示滚动区域，可使用左右方向键，Home 和 End 键" @scroll="measureScroll" @keydown="onKey">
      <div ref="canvas" class="diagram-canvas"><Mermaid :id="props.id" :graph="props.graph" /></div>
    </div>
    <div v-if="maxScroll > 1" class="diagram-navigation">
      <button type="button" aria-label="向左查看图示" :disabled="scrollLeft <= 1" @click="step(-1)">←</button>
      <input type="range" min="0" :max="maxScroll" :value="scrollLeft" aria-label="图示水平位置" @input="pan(Number(($event.target as HTMLInputElement).value))">
      <button type="button" aria-label="向右查看图示" :disabled="scrollLeft >= maxScroll - 1" @click="step(1)">→</button>
    </div>
    <figcaption>{{ maxScroll > 1 ? '左右滑动或拖动滑块查看完整图示，点击「全图」查看全貌。' : '可放大查看细节，点击百分比恢复原始大小。' }}</figcaption>
  </figure>
</template>
