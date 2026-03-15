const input = document.querySelector('input[type="file"]');
const previewBox = document.getElementById('previewBox');

if (input) {
  input.addEventListener('change', () => {
    const file = input.files[0];
    if (!file) {
      previewBox.textContent = 'No preview';
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewBox.innerHTML = `<img src="${e.target.result}" alt="preview">`;
    };
    reader.readAsDataURL(file);
  });
}
