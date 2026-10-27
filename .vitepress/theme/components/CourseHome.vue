<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { modules, lessonCount } from '../../curriculum'

const filters = [
  { id: 'all', label: '全部模块' },
  { id: 'foundation', label: '数据与工程' },
  { id: 'research', label: '因子与模型' },
  { id: 'portfolio', label: '组合与实盘' }
]
const activeFilter = ref('all')
const visibleModules = computed(() => modules.filter(module => activeFilter.value === 'all' || module.stage === activeFilter.value))
const stages = [
  { number: '01—02', label: '数据底座', detail: '工程栈 / 时点一致性', module: modules[0] },
  { number: '03—05', label: '信号研究', detail: '因子验证 / 机器学习', module: modules[2] },
  { number: '06—07', label: '组合执行', detail: '风险预算 / 交易约束', module: modules[5] },
  { number: '08', label: '归因监控', detail: '收益拆解 / 风险反馈', module: modules[7] }
]
// An in-page module link must reveal its target even after filtering.
function resetFilter() { activeFilter.value = 'all' }
</script>

<template>
  <main class="course-home">
    <section class="course-hero" aria-labelledby="course-title">
      <div class="hero-copy">
        <div class="hero-kicker"><span class="kicker-line" /> QUANTML <span class="kicker-divider">/</span> RESEARCH SERIES</div>
        <h1 id="course-title"><span>量化研究手册</span></h1>
        <p class="hero-title-en" lang="en">Quantitative Research Handbook</p>
        <p class="hero-subtitle">从数据、特征、AI 到指增实盘闭环</p>
        <p class="hero-description">以完整研究流程组织量化知识，将金融逻辑、数学方法与 Python 实践连接起来。</p>
        <div class="hero-actions">
          <a class="q-button primary" :href="withBase(`${modules[0].lessons[0].link}.html`)">开始第 1 讲 <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" /></svg></a>
          <a class="q-button secondary" href="#curriculum" @click="resetFilter">浏览课程体系 <span aria-hidden="true">↓</span></a>
        </div>
        <div class="hero-audience"><span class="audience-dot" />适合具备 Python 基础的学习者<span class="audience-separator">·</span>金融与数学系统进阶</div>
      </div>
      <div class="research-map" aria-label="量化研究流程">
        <div class="map-topline"><span>THE RESEARCH LOOP</span><span class="map-edition">A-SHARE / 01</span></div>
        <div class="map-heading"><h2>从研究到决策</h2><span>一条完整的投研路径</span></div>
        <ol class="research-stages">
          <li v-for="(stage, index) in stages" :key="stage.label">
            <a :href="withBase(`${stage.module.lessons[0].link}.html`)">
              <span class="stage-symbol" aria-hidden="true"><svg viewBox="0 0 24 24"><template v-if="index === 0"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v14c0 4 14 4 14 0V5M5 12c0 4 14 4 14 0"/></template><template v-else-if="index === 1"><path d="M3 18 8 11l5 3 8-10M3 21h18"/><circle cx="8" cy="11" r="1"/><circle cx="13" cy="14" r="1"/></template><template v-else-if="index === 2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></template><template v-else><path d="M3 12h4l3-7 4 14 3-7h4"/></template></svg></span>
              <span class="stage-copy"><strong>{{ stage.label }}</strong><span>{{ stage.detail }}</span></span>
              <span class="stage-number">{{ stage.number }}</span>
            </a>
          </li>
        </ol>
        <div class="map-bottomline"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9a8 8 0 0 1 14-3l2 3M20 3v6h-6M20 15A8 8 0 0 1 6 18l-2-3M4 21v-6h6"/></svg><span>验证 · 执行 · 归因 · 迭代</span><span class="map-dot" /></div>
      </div>
    </section>

    <section class="course-facts" aria-label="课程概览">
      <div class="course-fact"><strong>{{ String(modules.length).padStart(2, '0') }}</strong><span>核心模块<small>完整投研链路</small></span></div>
      <div class="course-fact"><strong>{{ lessonCount }}</strong><span>系统课程<small>由基础通向实践</small></span></div>
      <div class="course-principle"><span class="fact-marker">01 / REPRODUCIBLE</span><strong>可运行的研究实践</strong><span>Python 示例与方法验证</span></div>
      <div class="course-principle"><span class="fact-marker">02 / OPEN KNOWLEDGE</span><strong>开放的知识体系</strong><span>公开教材、论文与成熟工具</span></div>
    </section>

    <section id="curriculum" class="curriculum-section" aria-labelledby="curriculum-heading">
      <div class="section-heading"><div><span class="eyebrow">THE CURRICULUM</span><h2 id="curriculum-heading">构建完整的量化知识体系</h2></div><p>从数据可信，到信号有效，再到组合可执行。<br>沿研究流程逐步深入，也可按主题进入。</p></div>
      <div class="curriculum-toolbar">
        <div class="curriculum-filters" role="group" aria-label="按学习阶段筛选模块"><button v-for="filter in filters" :key="filter.id" type="button" :aria-pressed="activeFilter === filter.id" @click="activeFilter = filter.id">{{ filter.label }}</button></div>
        <span class="filter-count" aria-live="polite">{{ visibleModules.length }} 个模块 · {{ visibleModules.reduce((n, m) => n + m.lessons.length, 0) }} 讲</span>
      </div>
      <div class="module-grid">
        <article v-for="module in visibleModules" :id="`module-${module.id}`" :key="module.id" class="module-card">
          <div class="module-card-top"><span class="module-index">{{ module.id }}</span><span class="module-range">第 {{ module.lessons[0].number }}–{{ module.lessons[module.lessons.length - 1].number }} 讲</span></div>
          <span class="module-en">{{ module.en }}</span>
          <h3><a :href="withBase(`${module.lessons[0].link}.html`)">{{ module.title }}<span aria-hidden="true">↗</span></a></h3>
          <p>{{ module.description }}</p>
          <div class="module-tags"><span v-for="tag in module.tags" :key="tag">{{ tag }}</span></div>
          <details class="module-lessons"><summary><span>查看本模块课程</span><span class="lesson-toggle"><span>{{ module.lessons.length }} 讲</span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></span></summary>
            <ol><li v-for="lesson in module.lessons" :key="lesson.number"><a :href="withBase(`${lesson.link}.html`)"><span>{{ String(lesson.number).padStart(2, '0') }}</span><span>{{ lesson.title.replace(/^第 \d+ 讲：/, '') }}</span><span aria-hidden="true">↗</span></a></li></ol>
          </details>
        </article>
      </div>
    </section>

    <section class="learning-resources" aria-labelledby="resources-heading">
      <div class="resource-intro"><span class="eyebrow">BEYOND THE LESSONS</span><h2 id="resources-heading">让知识形成体系，<br>让研究可以复现。</h2><p>以课程为主线，以公开资源补足基础，<br>将每一讲落实为可检验的研究能力。</p></div>
      <div class="resource-links">
        <a :href="withBase('/learning_path.html')"><span class="resource-index">01</span><span><strong>学习路径</strong><small>入门自测、基础补充与逐章掌握标准</small></span><span aria-hidden="true">↗</span></a>
        <a :href="withBase('/resources.html')"><span class="resource-index">02</span><span><strong>公开资源导读</strong><small>大学公开课、经典教材、原始论文与官方文档</small></span><span aria-hidden="true">↗</span></a>
        <a :href="withBase('/README.html')"><span class="resource-index">03</span><span><strong>代码与环境</strong><small>配置本地环境，运行课程示例与测试</small></span><span aria-hidden="true">↗</span></a>
      </div>
    </section>
    <div class="toolchain"><span>RESEARCH TOOLKIT</span><span>Python</span><span>Polars</span><span>DuckDB</span><span>LightGBM</span><span>CVXPY</span></div>
  </main>
</template>
