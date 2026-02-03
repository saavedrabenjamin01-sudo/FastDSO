document.addEventListener('DOMContentLoaded', function() {
  const modal = document.getElementById('resultModal');
  if (!modal) return;

  const flashDataEl = document.getElementById('flashData');
  if (!flashDataEl) return;

  let flashData;
  try {
    flashData = JSON.parse(flashDataEl.textContent || '{}');
  } catch (e) {
    console.error('Error parsing flash data:', e);
    return;
  }

  if (!flashData.category || !flashData.message) return;

  const category = flashData.category;
  const message = flashData.message;
  const details = flashData.details || null;
  const primaryAction = flashData.primary_action || null;
  const primaryLabel = flashData.primary_label || 'Continuar';
  const secondaryLabel = flashData.secondary_label || 'Cargar otro';

  const iconSuccess = document.getElementById('resultIconSuccess');
  const iconError = document.getElementById('resultIconError');
  const iconWarning = document.getElementById('resultIconWarning');
  const iconInfo = document.getElementById('resultIconInfo');
  const resultCard = modal.querySelector('.result-card');
  const resultTitle = document.getElementById('resultTitle');
  const resultMsg = document.getElementById('resultMsg');
  const detailsWrap = document.getElementById('resultDetailsWrap');
  const detailsBody = document.getElementById('resultDetailsBody');
  const btnPrimary = document.getElementById('resultBtnPrimary');
  const btnSecondary = document.getElementById('resultBtnSecondary');

  iconSuccess.style.display = 'none';
  iconError.style.display = 'none';
  iconWarning.style.display = 'none';
  iconInfo.style.display = 'none';

  resultCard.classList.remove('result-card-success', 'result-card-danger', 'result-card-warning', 'result-card-info');

  if (category === 'success') {
    iconSuccess.style.display = 'block';
    resultCard.classList.add('result-card-success');
    resultTitle.textContent = 'Carga exitosa';
  } else if (category === 'danger' || category === 'error') {
    iconError.style.display = 'block';
    resultCard.classList.add('result-card-danger');
    resultTitle.textContent = 'Error en la carga';
  } else if (category === 'warning') {
    iconWarning.style.display = 'block';
    resultCard.classList.add('result-card-warning');
    resultTitle.textContent = 'Atención';
  } else {
    iconInfo.style.display = 'block';
    resultCard.classList.add('result-card-info');
    resultTitle.textContent = 'Información';
  }

  resultMsg.textContent = message;

  if (details) {
    detailsWrap.style.display = 'block';
    detailsBody.innerHTML = details;
  } else {
    detailsWrap.style.display = 'none';
  }

  btnPrimary.innerHTML = '<i class="bi bi-arrow-right me-1"></i>' + primaryLabel;
  if (primaryAction) {
    btnPrimary.href = primaryAction;
  } else {
    btnPrimary.style.display = 'none';
  }

  btnSecondary.innerHTML = '<i class="bi bi-plus-circle me-1"></i>' + secondaryLabel;
  btnSecondary.addEventListener('click', function() {
    modal.style.display = 'none';
    modal.classList.remove('result-show');
  });

  modal.style.display = 'flex';
  requestAnimationFrame(() => {
    modal.classList.add('result-show');
  });

  modal.addEventListener('click', function(e) {
    if (e.target === modal) {
      modal.style.display = 'none';
      modal.classList.remove('result-show');
    }
  });
});
