document.addEventListener('DOMContentLoaded', function() {
  var modal = document.getElementById('processingModal');
  if (!modal) return;

  var procTitle = document.getElementById('procTitle');
  var procMsg = document.getElementById('procMsg');
  var procPercent = document.getElementById('procPercent');
  var procProgressBar = modal.querySelector('.proc-progress-bar');
  var allIcons = modal.querySelectorAll('.proc-svg');

  var progressInterval = null;
  var currentProgress = 0;

  function showIcon(variant) {
    allIcons.forEach(function(icon) {
      icon.style.display = 'none';
    });
    var iconId = 'procIconGeneric';
    switch (variant) {
      case 'warehouse':
      case 'stock_cd':
        iconId = 'procIconStockCd';
        break;
      case 'store':
      case 'stock_store':
        iconId = 'procIconStockStore';
        break;
      case 'truck':
      case 'sales_dist':
        iconId = 'procIconSalesDist';
        break;
      case 'store-transfer':
      case 'redistribution':
        iconId = 'procIconRedistribution';
        break;
      default:
        iconId = 'procIconGeneric';
    }
    var icon = document.getElementById(iconId);
    if (icon) icon.style.display = 'block';
  }

  function updateProgress(percent) {
    currentProgress = percent;
    if (procProgressBar) {
      procProgressBar.style.width = percent + '%';
      procProgressBar.classList.remove('proc-progress-indeterminate');
    }
    if (procPercent) {
      procPercent.textContent = Math.floor(percent) + '%';
    }
  }

  function startSimulatedProgress() {
    currentProgress = 0;
    updateProgress(0);
    
    if (progressInterval) {
      clearInterval(progressInterval);
    }
    
    progressInterval = setInterval(function() {
      if (currentProgress < 70) {
        currentProgress += Math.random() * 3 + 1;
      } else if (currentProgress < 85) {
        currentProgress += Math.random() * 0.8 + 0.2;
      } else if (currentProgress < 95) {
        currentProgress += Math.random() * 0.3 + 0.05;
      }
      
      if (currentProgress > 95) {
        currentProgress = 95;
        clearInterval(progressInterval);
        progressInterval = null;
      }
      
      updateProgress(currentProgress);
    }, 120);
  }

  function stopProgress() {
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
  }

  function showModal(title, message, variant) {
    procTitle.textContent = title || 'Procesando';
    procMsg.textContent = message || 'Trabajando en tu solicitud...';
    showIcon(variant || 'generic');
    updateProgress(0);
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    startSimulatedProgress();
  }

  function hideModal() {
    stopProgress();
    modal.style.display = 'none';
    document.body.style.overflow = '';
  }

  document.querySelectorAll('[data-loader="true"]').forEach(function(el) {
    if (el.tagName === 'FORM') {
      el.addEventListener('submit', function(e) {
        var title = el.getAttribute('data-loader-title');
        var msg = el.getAttribute('data-loader-msg');
        var variant = el.getAttribute('data-loader-icon');
        
        el.querySelectorAll('button[type="submit"], input[type="submit"]').forEach(function(btn) {
          btn.disabled = true;
          if (btn.tagName === 'BUTTON') {
            btn.dataset.originalText = btn.innerHTML;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Procesando...';
          }
        });
        
        showModal(title, msg, variant);
      });
    } else if (el.tagName === 'BUTTON' || el.tagName === 'A') {
      el.addEventListener('click', function(e) {
        var title = el.getAttribute('data-loader-title');
        var msg = el.getAttribute('data-loader-msg');
        var variant = el.getAttribute('data-loader-icon');
        
        if (el.tagName === 'BUTTON') {
          el.disabled = true;
        }
        
        showModal(title, msg, variant);
      });
    }
  });

  window.showLoaderModal = showModal;
  window.hideLoaderModal = hideModal;
  window.updateLoaderProgress = updateProgress;
});
