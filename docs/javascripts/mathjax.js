window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex|md-content"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      // Re-typeset when details elements are opened (for quizzes)
      document.querySelectorAll('details').forEach(details => {
        details.addEventListener('toggle', () => {
          if (details.open) {
            MathJax.typesetPromise([details]);
          }
        });
      });
    }
  }
};

// For MkDocs Material instant loading
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
