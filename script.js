/* ═══════════════════════════════════════════════════════════
   MedPredict — Application Script
   Handles: Form Validation, Flask API Calls, Modal Display
   B.Tech Final Year Project | 2026
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── Navbar scroll effect ────────────────────────────────────
window.addEventListener('scroll', () => {
    const nav = document.getElementById('mainNav');
    if (!nav) return;
    if (window.scrollY > 60) {
        nav.style.background = 'rgba(2,11,24,0.97)';
    } else {
        nav.style.background = 'rgba(2,11,24,0.85)';
    }
});

// ── Smooth-close mobile menu after nav click ────────────────
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        const collapseEl = document.getElementById('navbarNav');
        if (collapseEl && collapseEl.classList.contains('show')) {
            const bsCollapse = bootstrap.Collapse.getInstance(collapseEl);
            if (bsCollapse) bsCollapse.hide();
        }
    });
});

// ── Main prediction handler ─────────────────────────────────
async function generateReport(type) {

    /* 1 ── Validate admin fields */
    const pName = document.getElementById('pName').value.trim();
    const lName = document.getElementById('lName').value.trim();

    if (!pName || !lName) {
        showValidationToast('Please enter Patient Name and Laboratory / Clinic Name before running the analysis.');
        document.getElementById('pName').focus();
        return;
    }

    /* 2 ── Validate form fields */
    const formId = type === 'heart' ? 'heartForm' : 'diabetesForm';
    const form   = document.getElementById(formId);

    if (!form) { console.error('Form not found:', formId); return; }

    const inputs = form.querySelectorAll('input[required]');
    let allFilled = true;

    inputs.forEach(input => {
        input.classList.remove('input-error');
        if (input.value === '' || input.value === null) {
            input.classList.add('input-error');
            allFilled = false;
        }
    });

    if (!allFilled) {
        showValidationToast('Please fill in all biomarker fields before running the analysis.');
        return;
    }

    /* 3 ── Open modal in loading state */
    const modal = new bootstrap.Modal(document.getElementById('reportModal'), { backdrop: 'static' });
    modal.show();
    setModalLoading(true);

    /* 4 ── Submit to Flask backend */
    const formData  = new FormData(form);
    const endpoint  = type === 'heart' ? '/predict_heart' : '/predict_diabetes';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body:   formData
        });

        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

        const data = await response.json();
        setModalLoading(false);

        if (data.status === 'success') {
            populateModal({
                name:        pName,
                lab:         lName,
                type:        type,
                result:      data.result,
                confidence:  data.confidence,
                xai:         data.xai
            });
        } else {
            showModalError(data.message || 'An unknown error occurred.');
        }

    } catch (err) {
        setModalLoading(false);
        console.error('Prediction error:', err);
        showModalError(
            'Could not connect to the Flask prediction server.<br>' +
            'Ensure <code>app.py</code> is running on <strong>localhost:5000</strong>.'
        );
    }
}

// ── Populate modal with prediction results ──────────────────
function populateModal({ name, lab, type, result, confidence, xai }) {

    /* Admin info */
    document.getElementById('outName').textContent  = name;
    document.getElementById('outLab').textContent   = lab;
    document.getElementById('outDate').textContent  = formatDate();
    document.getElementById('outModel').textContent = type === 'heart'
        ? 'Cardiac Risk Model (RF)'
        : 'Diabetes Screening Model (RF)';

    /* Result text */
    document.getElementById('outResult').textContent = result;

    /* Risk colouring */
    const alertEl = document.getElementById('resultAlert');
    const iconEl  = document.getElementById('outIcon');
    const isHigh  = /high|suspected/i.test(result);

    alertEl.className = `result-alert mb-4 ${isHigh ? 'high-risk' : 'low-risk'}`;

    iconEl.innerHTML = type === 'heart'
        ? `<i class="bi bi-heart-pulse-fill"></i>`
        : `<i class="bi bi-droplet-fill"></i>`;

    /* Confidence ring */
    const pctNum = parseFloat(confidence) || 0;
    document.getElementById('outConfidenceText').textContent = `${pctNum}%`;
    const circumference = 2 * Math.PI * 25; // r=25
    const offset = circumference - (pctNum / 100) * circumference;

    const circle = document.getElementById('confCircle');
    if (circle) {
        circle.style.strokeDasharray  = circumference;
        circle.style.strokeDashoffset = circumference; // start hidden
        setTimeout(() => { circle.style.strokeDashoffset = offset; }, 100);

        // Colour by risk level
        circle.style.stroke = isHigh ? '#f43f5e' : '#10b981';
    }

    /* XAI explanation */
    document.getElementById('outXAI').textContent = xai;
}

// ── Toggle loading state inside modal ──────────────────────
function setModalLoading(isLoading) {
    const loadingEl = document.getElementById('modalLoading');
    const resultEl  = document.getElementById('modalResult');
    if (!loadingEl || !resultEl) return;
    loadingEl.style.display = isLoading ? 'block' : 'none';
    resultEl.style.display  = isLoading ? 'none'  : 'block';
}

// ── Show error inside modal ─────────────────────────────────
function showModalError(msg) {
    const resultEl = document.getElementById('modalResult');
    resultEl.style.display = 'block';

    const alertEl = document.getElementById('resultAlert');
    alertEl.className = 'result-alert mb-4 high-risk';
    document.getElementById('outIcon').innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i>`;
    document.getElementById('outResult').textContent = 'Connection Error';
    document.getElementById('outXAI').innerHTML =
        `<span style="color:#fbbf24">${msg}</span>`;
    document.getElementById('outName').textContent = '—';
    document.getElementById('outLab').textContent  = '—';
    document.getElementById('outDate').textContent = '—';
    document.getElementById('outModel').textContent = '—';

    const circle = document.getElementById('confCircle');
    if (circle) circle.style.strokeDashoffset = 157;
    document.getElementById('outConfidenceText').textContent = '—';
}

// ── Inline validation toast ─────────────────────────────────
function showValidationToast(msg) {
    let toast = document.getElementById('valToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'valToast';
        toast.style.cssText = `
            position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
            background:#1e3a5f; border:1px solid rgba(14,165,233,0.4);
            color:#e2eaf5; padding:14px 24px; border-radius:12px;
            font-size:0.88rem; max-width:380px; text-align:center;
            box-shadow:0 8px 32px rgba(0,0,0,0.5);
            z-index:9999; opacity:0; transition:opacity 0.3s ease;
        `;
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { toast.style.opacity = '0'; }, 3500);
}

// ── Utility: format today's date ────────────────────────────
function formatDate() {
    return new Date().toLocaleDateString('en-IN', {
        day:   '2-digit',
        month: 'short',
        year:  'numeric'
    });
}

// ── Remove error highlighting on input ─────────────────────
document.addEventListener('input', e => {
    if (e.target.classList.contains('form-control-custom')) {
        e.target.classList.remove('input-error');
    }
});

// ── Inject input-error CSS once ────────────────────────────
(function injectErrorStyle() {
    const s = document.createElement('style');
    s.textContent = `.input-error { border-color:#f43f5e !important; box-shadow:0 0 0 3px rgba(244,63,94,0.15) !important; }`;
    document.head.appendChild(s);
})();
