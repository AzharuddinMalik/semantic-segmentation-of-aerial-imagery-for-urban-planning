const dropArea = document.getElementById("dropArea");
const fileElem = document.getElementById("fileElem");
const preview = document.getElementById("preview");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const removeFile = document.getElementById("removeFile");
const uploadForm = document.getElementById("uploadForm");

function handleFiles(files) {
    if (files.length) {
        const file = files[0];
        fileName.textContent = file.name;

        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        preview.classList.remove('hidden');
        dropArea.classList.add('hidden');
    }
}

removeFile.addEventListener("click", function() {
    fileElem.value = "";
    preview.classList.add('hidden');
    dropArea.classList.remove('hidden');
});

dropArea.addEventListener("dragover", function(e) {
    e.preventDefault();
    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", function() {
    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", function(e) {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    fileElem.files = e.dataTransfer.files;
    handleFiles(e.dataTransfer.files);
});

// Form submission handler
// Form submission handler
// Modify form submission handler
uploadForm.addEventListener("submit", async function(e) {
    e.preventDefault();
    const submitButton = uploadForm.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML;

    try {
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Processing...';
        submitButton.disabled = true;

        const formData = new FormData();
        formData.append('file', fileElem.files[0]);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || 'Upload failed');
        }

        const data = await response.json();

        // Update navigation to use base64 data
        window.location.href = `/result?input=${encodeURIComponent(data.input_image)}&mask=${encodeURIComponent(data.output_mask)}&confidence=${encodeURIComponent(data.confidence_map)}`;

    } catch (error) {
        console.error('Upload error:', error);
        alert(`Upload failed: ${error.message}`);
    } finally {
        submitButton.innerHTML = originalText;
        submitButton.disabled = false;
    }
});
