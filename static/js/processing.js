document.addEventListener('DOMContentLoaded', function() {
  var modal = document.getElementById('processingModal');
  if (!modal) return;

  var procTitle = document.getElementById('procTitle');
  var procMsg = document.getElementById('procMsg');
  var allIcons = modal.querySelectorAll('.proc-svg');

  function showIcon(variant) {
    allIcons.forEach(function(icon) {
      icon.style.display = 'none';
    });
    var iconId = 'procIconGeneric';
    switch (variant) {
      case 'stock_cd':
        iconId = 'procIconStockCd';
        break;
      case 'stock_store':
        iconId = 'procIconStockStore';
        break;
      case 'sales_dist':
        iconId = 'procIconSalesDist';
        break;
      case 'redistribution':
        iconId = 'procIconRedistribution';
        break;
      default:
        iconId = 'procIconGeneric';
    }
    var icon = document.getElementById(iconId);
    if (icon) icon.style.display = 'block';
  }

  function showModal(title, message, variant) {
    procTitle.textContent = title || 'Procesando';
    procMsg.textContent = message || 'Trabajando en tu solicitud...';
    showIcon(variant || 'generic');
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  function hideModal() {
    modal.style.display = 'none';
    document.body.style.overflow = '';
  }

  if (document.querySelector('.alert-danger') || document.querySelector('.alert-error')) {
    hideModal();
  }

  document.querySelectorAll('[data-processing="true"]').forEach(function(el) {
    if (el.tagName === 'FORM') {
      el.addEventListener('submit', function(e) {
        var title = el.getAttribute('data-proc-title');
        var msg = el.getAttribute('data-proc-msg');
        var variant = el.getAttribute('data-proc-variant');
        
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
        var title = el.getAttribute('data-proc-title');
        var msg = el.getAttribute('data-proc-msg');
        var variant = el.getAttribute('data-proc-variant');
        
        if (el.tagName === 'BUTTON') {
          el.disabled = true;
        }
        
        showModal(title, msg, variant);
      });
    }
  });

  window.showProcessingModal = showModal;
  window.hideProcessingModal = hideModal;
});
