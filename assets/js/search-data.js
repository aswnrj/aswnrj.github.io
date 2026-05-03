// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "Independent projects, mostly in ML systems and inference efficiency.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "post-t4-gpu-llama-why-your-attention-ooms-at-16k-and-the-one-line-fix",
        
          title: "T4 GPU + Llama: why your attention OOMs at 16K and the one-line...",
        
        description: "A walk through why PyTorch&#39;s SDPA falls through to a memory-blowup path on Turing GPUs, and what FlexAttention does differently.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/flashattention-t4/";
          
        },
      },{id: "projects-gpt-from-scratch",
          title: 'GPT from scratch',
          description: "A character-level GPT built from scratch in PyTorch — token embeddings, multi-head self-attention, feed-forward blocks, trained on tiny Shakespeare.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/gpt/";
            },},{id: "projects-bank-term-deposit-classification",
          title: 'Bank term-deposit classification',
          description: "Binary classification on a Kaggle bank-marketing dataset — Logistic Regression, XGBoost, LightGBM, and a stacked ensemble, evaluated with 5-fold stratified ROC-AUC.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/kaggle-bank-classification/";
            },},{id: "projects-multimodal-ai-search-engine",
          title: 'Multimodal AI search engine',
          description: "Semantic search across an image corpus using CLIP embeddings and FAISS — supports text-to-image and image-to-image queries with sub-second response times.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/multimodal-ai-search-engine/";
            },},{id: "projects-kv-cache-quantization-benchmark",
          title: 'KV-cache quantization benchmark',
          description: "Comparing FP16, INT8, and INT4 KV-cache compression on Llama 3.2-1B with HQQ — memory, throughput, and quality trade-offs.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/quantization-benchmark/";
            },},{id: "projects-sudoku-solver",
          title: 'Sudoku solver',
          description: "Browser-based Sudoku puzzle solver — interactive grid in JavaScript with a backtracking solver under the hood.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/sudoku-solver/";
            },},{id: "projects-to-do-list-app",
          title: 'To-do list app',
          description: "A full-stack to-do app with Node.js + Express, EJS templates, MongoDB via Mongoose, and Bootstrap on the frontend. Early CRUD project from when I was learning the stack.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/todolist-app/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%61%73%77%6E%72%6A.%72@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/aswnrj", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/aswnrj", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
