import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import Layout from './Layout.vue'
import CourseHome from './components/CourseHome.vue'
import CourseDiagram from './components/CourseDiagram.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('CourseHome', CourseHome)
    app.component('CourseDiagram', CourseDiagram)
  }
} satisfies Theme
