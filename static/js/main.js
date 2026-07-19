document.addEventListener('DOMContentLoaded', () => {
    // File upload preview logic
    const fileInput = document.getElementById('file-upload');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const uploadArea = document.querySelector('.upload-area');

    if (fileInput && previewContainer && imagePreview) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    previewContainer.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        // Drag and drop styles
        if(uploadArea) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    // Trigger change event manually
                    const event = new Event('change');
                    fileInput.dispatchEvent(event);
                }
            });
        }
    }
});
