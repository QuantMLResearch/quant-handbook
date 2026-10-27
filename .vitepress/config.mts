import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { modules } from './curriculum'

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: '量化研究手册',
    description: '从数据、特征、AI 到指增实盘闭环',
    
    appearance: 'dark',
    head: [['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }]],
    ignoreDeadLinks: true,

    // 启用内置数学公式 (MathJax) 与代码行号
    markdown: {
      math: true,
      lineNumbers: true,
      config(md) {
        const renderFence = md.renderer.rules.fence!
        md.renderer.rules.fence = (...args) => {
          const rendered = renderFence(...args)
          return args[0][args[1]].info.trim() === 'mermaid'
            ? rendered.replace(/<Mermaid\b/g, '<CourseDiagram').replace(/<\/Mermaid>/g, '</CourseDiagram>')
            : rendered
        }
        // Timeline uses ':' as a separator. Format clock punctuation for display
        // before Mermaid receives it, without changing the lesson source.
        md.core.ruler.after('block', 'timeline-clock-labels', state => {
          for (const token of state.tokens) {
            if (token.type === 'fence' && token.info.trim() === 'mermaid' && /^timeline\s/m.test(token.content)) {
              token.content = token.content.replace(/\b(\d{2}):(\d{2})\b/g, '$1：$2')
            }
          }
        })
      }
    },

    mermaid: {
      theme: 'base',
      themeVariables: {
        primaryColor: '#eaf2ff', primaryTextColor: '#182337', primaryBorderColor: '#2f80ed',
        lineColor: '#61748f', secondaryColor: '#e7f7f4', tertiaryColor: '#f3f6fb',
        fontFamily: 'IBM Plex Sans, Noto Sans SC, sans-serif'
      },
      flowchart: { useMaxWidth: false },
      sequence: { useMaxWidth: false },
      timeline: { useMaxWidth: false }
    },

    vite: {
      optimizeDeps: {
        include: ['mermaid', 'fastdom']
      }
    },

    themeConfig: {
      siteTitle: 'QuantML',
      logo: { src: '/favicon.svg', alt: '' },
      sidebarMenuLabel: '课程目录',
      returnToTopLabel: '返回顶部',
      skipToContentLabel: '跳至正文',
      darkModeSwitchLabel: '外观',
      darkModeSwitchTitle: '切换至深色模式',
      lightModeSwitchTitle: '切换至浅色模式',
      nav: [
        { text: '课程总览', link: '/#curriculum' },
        { text: '学习路径', link: '/learning_path' },
        { text: '公开资源', link: '/resources' },
        { text: '课程大纲', link: '/syllabus' },
        { text: '官网', link: 'https://www.quantml.cn/' }
      ],
      sidebar: [
        {
          text: '学习指南',
          collapsed: true,
          items: [
            { text: '学习路径与掌握标准', link: '/learning_path' },
            { text: '公开资源导读', link: '/resources' },
            { text: '课程大纲', link: '/syllabus' }
          ]
        },
        ...modules.map(module => ({
          text: `${module.id} · ${module.title}`,
          collapsed: true,
          items: module.lessons.map(lesson => ({ text: lesson.title, link: lesson.link }))
        }))
      ],
      search: {
        provider: 'local',
        options: {
          locales: {
            root: {
              translations: {
                button: {
                  buttonText: '搜索课程',
                  buttonAriaLabel: '搜索课程'
                },
                modal: {
                  noResultsText: '无法找到相关结果',
                  resetButtonTitle: '清除查询条件',
                  footer: {
                    selectText: '选择',
                    navigateText: '切换',
                    closeText: '关闭'
                  }
                }
              }
            }
          }
        }
      },
      outline: {
        level: [2, 3],
        label: '本讲大纲'
      },
      docFooter: {
        prev: '上一讲',
        next: '下一讲'
      },
      footer: {
        message: 'QUANTML · Quantitative Research Handbook',
        copyright: 'Copyright © 2026 量化研究手册'
      }
    }
  })
)
