<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { modules, lessonCount } from '../curriculum'

const { page, frontmatter } = useData()
const currentModule = computed(() => modules.find(module => module.lessons.some(
  lesson => `${lesson.link.slice(1)}.md` === page.value.relativePath
)))
const currentLesson = computed(() => currentModule.value?.lessons.find(
  lesson => `${lesson.link.slice(1)}.md` === page.value.relativePath
))
</script>

<template>
  <DefaultTheme.Layout>
    <template #nav-bar-title-after><span class="brand-caption">量化研究手册</span></template>
    <template #sidebar-nav-before>
      <a class="sidebar-heading" :href="withBase('/#curriculum')">
        <span class="eyebrow">COURSE HANDBOOK</span>
        <span>课程目录 <span class="sidebar-count">{{ modules.length }} 模块 / {{ lessonCount }} 讲</span></span>
      </a>
    </template>
    <template #doc-before>
      <div v-if="currentModule && currentLesson" class="lesson-context">
        <nav class="lesson-breadcrumb" aria-label="当前位置">
          <a :href="withBase('/#curriculum')">课程总览</a>
          <span aria-hidden="true">/</span>
          <a :href="withBase(`/#module-${currentModule.id}`)">{{ currentModule.title }}</a>
        </nav>
        <div class="lesson-meta">
          <span class="lesson-number">LECTURE {{ String(currentLesson.number).padStart(2, '0') }}</span>
          <span>共 {{ lessonCount }} 讲</span>
          <span v-if="frontmatter.reading_time" class="reading-time">预计阅读 {{ frontmatter.reading_time }}</span>
        </div>
      </div>
      <div v-else class="guide-context"><span class="eyebrow">QUANTML / LEARNING GUIDE</span></div>
    </template>
    <template #aside-outline-before>
      <div v-if="currentModule" class="aside-module">
        <span class="eyebrow">MODULE {{ currentModule.id }}</span>
        <strong>{{ currentModule.title }}</strong>
        <span>{{ currentModule.lessons.length }} 讲 · {{ currentModule.en }}</span>
      </div>
    </template>
    <template #aside-outline-after>
      <div class="aside-resources">
        <span>配套学习资料</span>
        <a :href="withBase('/resources.html')">公开教材与工具文档 <span aria-hidden="true">↗</span></a>
        <a :href="withBase('/learning_path.html')">学习路径与掌握标准 <span aria-hidden="true">↗</span></a>
      </div>
    </template>
  </DefaultTheme.Layout>
</template>
