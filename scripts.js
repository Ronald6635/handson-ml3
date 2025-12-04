// Dark mode toggle functionality
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    updatePrismTheme();
    // Save preference to localStorage
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
}

// Load dark mode preference on page load
document.addEventListener('DOMContentLoaded', function() {
    const darkMode = localStorage.getItem('darkMode');
    if (darkMode === 'true') {
        document.body.classList.add('dark-mode');
        updatePrismTheme();
    }
});

// Back to top functionality
window.onscroll = function() {showBackToTopButton()};

function showBackToTopButton() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        document.querySelector('.back-to-top').style.display = "block";
    } else {
        document.querySelector('.back-to-top').style.display = "none";
    }
}

function scrollToTop() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
}

// Mobile menu toggle
function toggleMobileMenu() {
    const navMenu = document.querySelector('.nav-menu');
    navMenu.classList.toggle('mobile-visible');
}

// Search functionality
function performSearch() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase().trim();
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '';

    if (searchTerm === '') {
        return;
    }

    const sections = document.querySelectorAll('section[id^="chapter-"], section[id="glossary"], section[id="appendices"]');
    let resultsFound = false;

    sections.forEach(section => {
        const sectionTitle = section.querySelector('h2').textContent.toLowerCase();
        const tables = section.querySelectorAll('table');
        let sectionMatches = [];

        // Check section title
        if (sectionTitle.includes(searchTerm)) {
            sectionMatches.push({
                type: 'section',
                title: section.querySelector('h2').textContent,
                link: `#${section.id}`
            });
        }

        // Check table content
        tables.forEach((table, tableIndex) => {
            const rows = table.querySelectorAll('tr');
            rows.forEach((row, rowIndex) => {
                const cells = row.querySelectorAll('td, th');
                cells.forEach(cell => {
                    const cellText = cell.textContent.toLowerCase();
                    if (cellText.includes(searchTerm)) {
                        // Find the concept/function name (usually first column)
                        const firstCell = row.querySelector('td:first-child, th:first-child');
                        if (firstCell) {
                            sectionMatches.push({
                                type: 'table-entry',
                                title: firstCell.textContent.trim(),
                                section: section.querySelector('h2').textContent,
                                link: `#${section.id}`
                            });
                        }
                    }
                });
            });
        });

        // Display results for this section
        if (sectionMatches.length > 0) {
            resultsFound = true;
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'search-result-section';

            const sectionHeader = document.createElement('h3');
            sectionHeader.textContent = section.querySelector('h2').textContent;
            sectionDiv.appendChild(sectionHeader);

            sectionMatches.forEach(match => {
                const resultItem = document.createElement('div');
                resultItem.className = 'search-result-item';

                const link = document.createElement('a');
                link.href = match.link;
                link.textContent = match.title;
                if (match.type === 'table-entry') {
                    link.textContent += ` (${match.section})`;
                }

                resultItem.appendChild(link);
                sectionDiv.appendChild(resultItem);
            });

            resultsContainer.appendChild(sectionDiv);
        }
    });

    if (!resultsFound) {
        resultsContainer.innerHTML = '<p class="no-results">未找到匹配的結果。請嘗試不同的關鍵字。</p>';
    }
}

function clearSearch() {
    document.getElementById('search-input').value = '';
    document.getElementById('search-results').innerHTML = '';
}

// Event listeners
document.getElementById('search-button').addEventListener('click', performSearch);
document.getElementById('clear-search').addEventListener('click', clearSearch);
document.getElementById('search-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Copy code functionality
function copyCode(button) {
    const codeElement = button.previousElementSibling;
    const codeText = codeElement.textContent || codeElement.innerText;

    navigator.clipboard.writeText(codeText).then(function() {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        button.style.backgroundColor = '#28a745';
        setTimeout(function() {
            button.textContent = originalText;
            button.style.backgroundColor = '';
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = codeText;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.style.backgroundColor = '#28a745';
            setTimeout(function() {
                button.textContent = originalText;
                button.style.backgroundColor = '';
            }, 2000);
        } catch (err) {
            console.error('Fallback copy failed: ', err);
            button.textContent = 'Copy failed';
            button.style.backgroundColor = '#dc3545';
            setTimeout(function() {
                button.textContent = 'Copy';
                button.style.backgroundColor = '';
            }, 2000);
        }
        document.body.removeChild(textArea);
    });
}

// Collapsible sections functionality
function toggleSection(button) {
    const section = button.closest('section');
    const content = section.querySelector('.section-content');
    const isExpanded = button.getAttribute('aria-expanded') === 'true';

    if (isExpanded) {
        content.style.display = 'none';
        button.setAttribute('aria-expanded', 'false');
        button.textContent = '展開';
    } else {
        content.style.display = 'block';
        button.setAttribute('aria-expanded', 'true');
        button.textContent = '收起';
    }
}

// Initialize collapsible sections
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('section[id^="chapter-"]');
    sections.forEach(section => {
        const h2 = section.querySelector('h2');
        if (h2) {
            // Add toggle button
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'section-toggle';
            toggleBtn.setAttribute('aria-expanded', 'true');
            toggleBtn.setAttribute('aria-controls', section.id + '-content');
            toggleBtn.textContent = '收起';
            toggleBtn.onclick = function() { toggleSection(this); };
            toggleBtn.onkeydown = function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleSection(this);
                }
            };
            toggleBtn.tabIndex = 0;

            // Wrap content in a div
            const content = document.createElement('div');
            content.className = 'section-content';
            content.id = section.id + '-content';

            // Move all children except h2 into content
            while (section.children.length > 1) {
                content.appendChild(section.children[1]);
            }

            h2.appendChild(toggleBtn);
            section.appendChild(content);
        }
    });
});

// Update Prism theme based on dark mode
function updatePrismTheme() {
    const darkTheme = document.getElementById('prism-dark-theme');
    if (document.body.classList.contains('dark-mode')) {
        darkTheme.disabled = false;
    } else {
        darkTheme.disabled = true;
    }
}

// Override toggleDarkMode to also update Prism theme
const originalToggleDarkMode = toggleDarkMode;
toggleDarkMode = function() {
    originalToggleDarkMode();
    updatePrismTheme();
};

// Initialize Prism theme
updatePrismTheme();

// Feedback functionality
function openFeedback() {
    const feedbackUrl = 'https://github.com/Ronald6635/handson-ml3/issues/new?title=Feedback%20for%20ML%20Cheat%20Sheet&body=Please%20describe%20your%20feedback%20or%20suggestions%20for%20the%20machine%20learning%20cheat%20sheet.';
    window.open(feedbackUrl, '_blank');
}

// Error handling for MathJax loading
window.addEventListener('error', function(e) {
    if (e.target.tagName === 'SCRIPT' && e.target.src.includes('mathjax')) {
        console.warn('MathJax failed to load. Mathematical expressions may not render properly.');
        // Could show a user notification here
    }
});

// Service worker registration for offline access
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}