const form = document.getElementById('predict-form');
const resultEl = document.getElementById('result');

if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    for (const [k, v] of Object.entries(payload)) {
      if (v === '' || isNaN(Number(v))) {
        showResult(false, `Please enter a numeric value for "${k}".`);
        return;
      }
    }

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Accept': 'application/json' },
        body: formData
      });
      const data = await res.json();
      if (!data.ok) {
        showResult(false, data.error || 'Prediction failed.');
        return;
      }

      const msg = data.has_parkinsons
        ? `Prediction: Parkinson's LIKELY (status=1)` + (data.probability != null ? ` | Confidence ≈ ${Math.round(data.probability*100)}%` : '')
        : `Prediction: Parkinson's UNLIKELY (status=0)` + (data.probability != null ? ` | Confidence ≈ ${Math.round((1-data.probability)*100)}%` : '');

      showResult(true, msg);
    } catch (err) {
      showResult(false, err.message || String(err));
    }
  });
}

function showResult(ok, text) {
  resultEl.classList.remove('hidden', 'ok', 'err');
  resultEl.classList.add(ok ? 'ok' : 'err');
  resultEl.textContent = text;
}
