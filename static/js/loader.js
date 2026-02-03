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

  function updateMessage(msg) {
    if (procMsg) {
      procMsg.textContent = msg;
    }
  }

  function showErrorAlert(message, traceback) {
    var existingAlert = document.getElementById('ajax-error-alert');
    if (existingAlert) existingAlert.remove();
    
    var alertHtml = '<div id="ajax-error-alert" class="alert alert-danger alert-dismissible fade show" role="alert">' +
      '<strong>Error:</strong> ' + message;
    
    if (traceback) {
      alertHtml += '<details class="mt-2"><summary>Detalles técnicos</summary>' +
        '<pre class="small mb-0 mt-2" style="max-height:200px;overflow:auto;">' + traceback + '</pre></details>';
    }
    
    alertHtml += '<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>';
    
    var mainContent = document.querySelector('.container-fluid') || document.querySelector('main') || document.body;
    mainContent.insertAdjacentHTML('afterbegin', alertHtml);
    
    window.scrollTo(0, 0);
  }

  function resetFormButtons(form) {
    if (!form) return;
    form.querySelectorAll('button[type="submit"], input[type="submit"]').forEach(function(btn) {
      btn.disabled = false;
      if (btn.tagName === 'BUTTON' && btn.dataset.originalText) {
        btn.innerHTML = btn.dataset.originalText;
      }
    });
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
    var card = modal.querySelector('.proc-card');
    card.classList.remove('proc-card-success', 'proc-card-error');
    
    var existingActions = card.querySelector('.proc-result-actions');
    if (existingActions) existingActions.remove();
    
    var progressWrap = modal.querySelector('.proc-progress');
    var percentEl = document.getElementById('procPercent');
    if (progressWrap) progressWrap.style.display = '';
    if (percentEl) percentEl.style.display = '';
    
    procTitle.textContent = title || 'Procesando';
    procMsg.textContent = message || 'Trabajando en tu solicitud...';
    showIcon(variant || 'generic');
    updateProgress(0);
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  function hideModal() {
    stopProgress();
    modal.style.display = 'none';
    document.body.style.overflow = '';
  }

  function showSuccessState(title, message, redirectUrl) {
    stopProgress();
    var card = modal.querySelector('.proc-card');
    card.classList.add('proc-card-success');
    
    allIcons.forEach(function(icon) { icon.style.display = 'none'; });
    
    var iconContainer = document.getElementById('procIcon');
    iconContainer.innerHTML = '<svg class="proc-svg proc-svg-success" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">' +
      '<circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="3" fill="none"/>' +
      '<path class="proc-check-anim" d="M20 32L28 40L44 24" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>' +
      '</svg>';
    
    procTitle.textContent = title;
    procMsg.textContent = message;
    
    var progressWrap = modal.querySelector('.proc-progress');
    var percentEl = document.getElementById('procPercent');
    if (progressWrap) progressWrap.style.display = 'none';
    if (percentEl) percentEl.style.display = 'none';
    
    var actionsHtml = '<div class="proc-result-actions">' +
      '<a href="' + redirectUrl + '" class="btn proc-btn-primary"><i class="bi bi-arrow-right me-2"></i>Continuar</a>' +
      '<button type="button" class="btn proc-btn-secondary" onclick="window.location.reload()"><i class="bi bi-plus-circle me-2"></i>Cargar otro</button>' +
      '</div>';
    
    var existingActions = card.querySelector('.proc-result-actions');
    if (existingActions) existingActions.remove();
    card.insertAdjacentHTML('beforeend', actionsHtml);
  }

  function showErrorState(title, message, traceback) {
    stopProgress();
    var card = modal.querySelector('.proc-card');
    card.classList.add('proc-card-error');
    
    allIcons.forEach(function(icon) { icon.style.display = 'none'; });
    
    var iconContainer = document.getElementById('procIcon');
    iconContainer.innerHTML = '<svg class="proc-svg proc-svg-error" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">' +
      '<circle cx="32" cy="32" r="28" stroke="currentColor" stroke-width="3" fill="none"/>' +
      '<path d="M22 22L42 42M42 22L22 42" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>' +
      '</svg>';
    
    procTitle.textContent = title;
    procMsg.textContent = message;
    
    var progressWrap = modal.querySelector('.proc-progress');
    var percentEl = document.getElementById('procPercent');
    if (progressWrap) progressWrap.style.display = 'none';
    if (percentEl) percentEl.style.display = 'none';
    
    var detailsHtml = '';
    if (traceback) {
      detailsHtml = '<details class="proc-error-details"><summary>Detalles técnicos</summary>' +
        '<pre class="proc-error-traceback">' + traceback + '</pre></details>';
    }
    
    var actionsHtml = '<div class="proc-result-actions">' + detailsHtml +
      '<button type="button" class="btn proc-btn-secondary" onclick="window.location.reload()"><i class="bi bi-arrow-clockwise me-2"></i>Intentar de nuevo</button>' +
      '</div>';
    
    var existingActions = card.querySelector('.proc-result-actions');
    if (existingActions) existingActions.remove();
    card.insertAdjacentHTML('beforeend', actionsHtml);
  }

  function pollJobStatus(jobId, redirectUrl, variant) {
    updateMessage('Procesando datos en el servidor...');
    
    function poll() {
      fetch('/jobs/' + jobId)
        .then(function(r) { return r.json(); })
        .then(function(data) {
          var jobProgress = Math.min(data.progress || 0, 100);
          var displayProgress = 50 + (jobProgress * 0.5);
          updateProgress(displayProgress);
          
          if (data.message) {
            updateMessage(data.message);
          }
          
          if (data.status === 'done') {
            updateProgress(100);
            showSuccessState(
              data.result_title || 'Carga completada',
              data.result_message || data.message || 'El proceso finalizó correctamente.',
              redirectUrl
            );
          } else if (data.status === 'error') {
            showErrorState(
              'Error en el proceso',
              data.message || 'Error desconocido',
              data.traceback
            );
          } else {
            setTimeout(poll, 1500);
          }
        })
        .catch(function(err) {
          console.error('Poll error:', err);
          setTimeout(poll, 3000);
        });
    }
    
    poll();
  }

  function submitFormWithXHR(form, title, msg, variant) {
    var formData = new FormData(form);
    var xhr = new XMLHttpRequest();
    
    showModal(title, msg, variant);
    
    xhr.upload.onprogress = function(e) {
      if (e.lengthComputable) {
        var percent = (e.loaded / e.total) * 50;
        updateProgress(percent);
        if (percent < 50) {
          updateMessage('Subiendo archivo... ' + Math.floor((e.loaded / e.total) * 100) + '%');
        }
      }
    };
    
    xhr.onload = function() {
      if (xhr.status === 200) {
        try {
          var resp = JSON.parse(xhr.responseText);
          if (resp.ok && resp.job_id) {
            updateProgress(50);
            updateMessage('Archivo recibido. Procesando...');
            pollJobStatus(resp.job_id, resp.redirect_url || '/dashboard', variant);
          } else if (resp.ok && resp.redirect_url) {
            updateProgress(100);
            window.location.href = resp.redirect_url;
          } else {
            hideModal();
            showErrorAlert(resp.message || 'Error desconocido', resp.traceback);
            resetFormButtons(form);
          }
        } catch (e) {
          hideModal();
          showErrorAlert('Error procesando respuesta del servidor');
          resetFormButtons(form);
        }
      } else {
        hideModal();
        try {
          var errResp = JSON.parse(xhr.responseText);
          showErrorAlert(errResp.message || 'Error en el servidor', errResp.traceback);
        } catch (e) {
          showErrorAlert('Error en el servidor: ' + xhr.status);
        }
        resetFormButtons(form);
      }
    };
    
    xhr.onerror = function() {
      hideModal();
      showErrorAlert('Error de conexión. Intenta nuevamente.');
      resetFormButtons(form);
    };
    
    xhr.open('POST', form.action || window.location.href, true);
    xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
    xhr.send(formData);
  }

  document.querySelectorAll('[data-loader="true"]').forEach(function(el) {
    if (el.tagName === 'FORM') {
      el.addEventListener('submit', function(e) {
        var title = el.getAttribute('data-loader-title');
        var msg = el.getAttribute('data-loader-msg');
        var variant = el.getAttribute('data-loader-icon');
        var isAjax = el.getAttribute('data-ajax') === 'true';
        
        el.querySelectorAll('button[type="submit"], input[type="submit"]').forEach(function(btn) {
          btn.disabled = true;
          if (btn.tagName === 'BUTTON') {
            btn.dataset.originalText = btn.innerHTML;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Procesando...';
          }
        });
        
        if (isAjax) {
          e.preventDefault();
          submitFormWithXHR(el, title, msg, variant);
        } else {
          showModal(title, msg, variant);
          startSimulatedProgress();
        }
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
        startSimulatedProgress();
      });
    }
  });

  window.showLoaderModal = showModal;
  window.hideLoaderModal = hideModal;
  window.updateLoaderProgress = updateProgress;
  window.updateLoaderMessage = updateMessage;
});
