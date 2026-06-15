document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingUI = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    
    // Drag & Drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleUpload(e.target.files[0]);
        }
    });

    async function handleUpload(file) {
        // UI Reset
        dropZone.classList.add('hidden');
        loadingUI.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        // Fake progress animation for pipeline steps
        simulateProgress();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (!response.ok) throw new Error(data.error || 'Server error');
            
            renderResults(data);
        } catch (error) {
            alert('Pipeline Error: ' + error.message);
            dropZone.classList.remove('hidden');
            loadingUI.classList.add('hidden');
        }
    }

    function renderResults(data) {
        loadingUI.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        dropZone.classList.remove('hidden'); // allow another upload

        document.getElementById('res-raw').src = '/' + data.raw_image;
        document.getElementById('res-bin').src = '/' + data.binarized_image;
        document.getElementById('glyph-count').innerText = data.num_glyphs;

        const container = document.getElementById('clusters-container');
        container.innerHTML = '';

        if(data.clusters.length === 0) {
            container.innerHTML = '<p style="color:var(--text-secondary)">No clusters detected.</p>';
            return;
        }

        data.clusters.forEach(cluster => {
            const card = document.createElement('div');
            card.className = 'cluster-card';
            
            // Format name nicely
            const title = cluster.name.replace('_', ' ');
            
            card.innerHTML = `
                <h3>${title} <span>[${cluster.images.length} items]</span></h3>
                <div class="glyph-grid">
                    ${cluster.images.map(img => `<img src="/${img}" class="glyph-item" loading="lazy">`).join('')}
                </div>
            `;
            container.appendChild(card);
        });
    }

    function simulateProgress() {
        const steps = [
            document.getElementById('step-1'),
            document.getElementById('step-2'),
            document.getElementById('step-3'),
            document.getElementById('step-4')
        ];
        
        steps.forEach(s => s.classList.remove('active'));
        
        let current = 0;
        const interval = setInterval(() => {
            if(current < steps.length) {
                steps[current].classList.add('active');
                current++;
            } else {
                clearInterval(interval);
            }
        }, 1200); // Activate a step every 1.2s to match expected processing time
    }
});
